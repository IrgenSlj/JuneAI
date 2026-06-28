"""Regression tests for setup runtime reloading."""

from __future__ import annotations

from june_api.routes import setup as setup_route
from june_api.schemas import SetupApplyRequest
from june_brain.config import RuntimeConfig
from june_brain.config_store import StoredConfig


def test_apply_setup_verifies_provider(monkeypatch) -> None:
    monkeypatch.setattr(setup_route, "load_stored_config", lambda: StoredConfig())
    monkeypatch.setattr(setup_route, "save_stored_config", lambda _stored: None)
    monkeypatch.setattr(setup_route, "is_ollama_running", lambda _base_url: True)
    monkeypatch.setattr(
        setup_route,
        "is_model_available",
        lambda _model, _base_url: True,
    )
    # The verification round-trip goes through the provider stack; stub it green
    # so the apply flow can be tested without a live model.
    monkeypatch.setattr(setup_route, "_verify_round_trip", lambda _runtime: (True, "", ""))

    response = setup_route.apply_setup(SetupApplyRequest(provider="gemma"))

    # No agent to invalidate/reload (ADR 0018): _verify_round_trip is the check.
    assert response.ok is True
    assert response.verified is True


def test_setup_status_marks_semantic_recall_degraded_when_embed_model_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(setup_route, "load_stored_config", lambda: StoredConfig(provider="gemma"))
    monkeypatch.setattr(setup_route, "is_configured", lambda _stored: True)
    monkeypatch.setattr(setup_route, "default_embed_model", lambda: "test-embed")
    monkeypatch.setattr(setup_route, "is_ollama_running", lambda _base_url: True)

    def _available(model: str, _base_url: str) -> bool:
        return model == "gemma4:e2b"

    monkeypatch.setattr(setup_route, "is_model_available", _available)
    monkeypatch.setattr(
        setup_route,
        "resolve_runtime_config",
        lambda: RuntimeConfig(
            preset_key="gemma",
            label="Gemma 4 (local)",
            provider="openai_compatible",
            model="gemma4:e2b",
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            temperature=0.4,
            max_tokens=4096,
            tool_strategy="native",
            prompt_style="gemma",
        ),
    )

    response = setup_route.get_setup_status()

    assert response.is_configured is True
    assert response.ollama_reachable is True
    assert response.ollama_has_model is True
    assert response.embedding_model == "test-embed"
    assert response.embedding_available is False
    assert response.semantic_recall_status == "degraded"
    assert "keyword search" in response.semantic_recall_detail
