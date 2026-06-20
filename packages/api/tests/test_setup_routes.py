"""Regression tests for setup runtime reloading."""

from __future__ import annotations

from june_api.routes import setup as setup_route
from june_api.schemas import SetupApplyRequest
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
