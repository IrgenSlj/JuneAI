"""First-run setup endpoints.

These power the /setup screen in the web UI. The goal is to turn a fresh
install into a working configuration with one round-trip to a real model,
so the user never sees a broken state after completing the form.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from june_brain.config import DEFAULT_RUNTIME_PRESET, resolve_runtime_config
from june_brain.config_store import (
    StoredConfig,
    is_configured,
    load_stored_config,
    save_stored_config,
)
from june_brain.memory.embeddings import default_embed_model
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
            user_name=stored.user_name or "",
        )

    ollama_reachable = False
    ollama_has_model = False
    embedding_model = default_embed_model()
    embedding_available = False
    semantic_recall_status = "unknown"
    semantic_recall_detail = "Semantic recall readiness was not checked for this runtime."
    if runtime.preset_key == "gemma":
        ollama_reachable = is_ollama_running(runtime.base_url)
        if ollama_reachable:
            ollama_has_model = is_model_available(runtime.model, runtime.base_url)
            embedding_available = is_model_available(embedding_model, runtime.base_url)
            if embedding_available:
                semantic_recall_status = "ready"
                semantic_recall_detail = (
                    f"Embedding model {embedding_model!r} is available for semantic recall."
                )
            else:
                semantic_recall_status = "degraded"
                semantic_recall_detail = (
                    f"Embedding model {embedding_model!r} is not pulled; semantic recall "
                    "falls back to SQLite keyword search."
                )
        else:
            semantic_recall_status = "degraded"
            semantic_recall_detail = (
                "Ollama is not reachable; semantic recall falls back to SQLite keyword search."
            )

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
        embedding_model=embedding_model,
        embedding_available=embedding_available,
        semantic_recall_status=semantic_recall_status,
        semantic_recall_detail=semantic_recall_detail,
        api_key_present=api_key_present,
        user_name=stored.user_name or "",
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
    if request.user_name:
        stored.user_name = request.user_name
    if request.privacy_dial:
        stored.privacy_dial = request.privacy_dial
    if request.gemma_model:
        stored.gemma_model = request.gemma_model
    if request.gemini_model:
        stored.gemini_model = request.gemini_model
    if request.gemini_api_key:
        stored.gemini_api_key = request.gemini_api_key.strip()

    save_stored_config(stored)
    _apply_to_env(stored)
    # No agent to invalidate: the hand-written loop resolves config and keys
    # fresh each turn (ADR 0018). _verify_round_trip below is the real check.

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


def _provider_for_runtime(runtime):
    """Build the concrete provider for a runtime, ready for a one-shot generate.

    Uses the same provider classes the loop uses (no LangChain). The cloud
    provider resolves its key from env/keyring at call time, so the key applied
    earlier in apply_setup is picked up here.
    """
    from june_brain.providers.gemini import GeminiProvider
    from june_brain.providers.gemma import GemmaProvider

    if runtime.is_api:
        return GeminiProvider(model_id=runtime.model, base_url=runtime.base_url)
    return GemmaProvider(
        model_id=runtime.model, base_url=runtime.base_url, tier="local-fast"
    )


def _verify_round_trip(runtime) -> tuple[bool, str, str]:
    """Ask the configured model to reply once. Any response at all counts as verified."""
    import asyncio

    from june_brain.providers.base import GenerateRequest, Message

    try:
        provider = _provider_for_runtime(runtime)
        result = asyncio.run(
            provider.generate(
                GenerateRequest(
                    messages=[
                        Message(role="user", content="Reply with the single word OK.")
                    ],
                    max_tokens=16,
                )
            )
        )
    except Exception as exc:  # noqa: BLE001 — the UI needs a message, not a stack trace
        return False, f"Verification failed: {exc}", _error_hint(runtime, exc)

    if not str(result.text).strip():
        return False, "Provider returned an empty response.", "Try a different model tag."
    return True, "", ""


_GENERIC_HINT = "Check your network and credentials, then try again."

# Hints for the verify round-trip, keyed by preset. Each entry is a list of
# (substring-match, hint) pairs evaluated in order; the first match wins.
# Centralised here so a vendor changing its error wording is a one-line edit
# and so the set of error cases is reviewable in one place.
_ERROR_HINTS: dict[str, list[tuple[str, str]]] = {
    "gemini": [
        ("401", "The Gemini API key looks invalid. Generate a new one at https://aistudio.google.com."),
        ("api key", "The Gemini API key looks invalid. Generate a new one at https://aistudio.google.com."),
        ("permission", "The Gemini API key looks invalid. Generate a new one at https://aistudio.google.com."),
        ("quota", "Gemini quota exceeded for this key. Try again later or use a different key."),
        ("rate limit", "Gemini rate-limited this key. Wait a minute and retry."),
    ],
    "gemma": [
        ("connection", "Ollama isn't accepting connections. Is `ollama serve` running?"),
        ("refused", "Ollama isn't accepting connections. Is `ollama serve` running?"),
        ("not found", "The model tag isn't pulled. Run `ollama pull <model>` and retry."),
        ("timeout", "Ollama took too long to respond. The model may still be loading — retry in a moment."),
    ],
}


def _error_hint(runtime, exc: Exception) -> str:
    message = str(exc).lower()
    for needle, hint in _ERROR_HINTS.get(runtime.preset_key, []):
        if needle in message:
            return hint
    return _GENERIC_HINT
