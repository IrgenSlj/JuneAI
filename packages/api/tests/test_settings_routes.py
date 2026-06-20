"""Regression tests for settings runtime invalidation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from june_api.routes import settings as settings_route
from june_api.schemas import PrivacyDialUpdateRequest


def test_forget_key_scrubs_env(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    monkeypatch.setattr(settings_route, "forget_gemini_key", lambda: "file")

    response = settings_route.forget_key()

    assert response.cleared_from == "file"
    # No agent to invalidate: the provider re-resolves its key on the next call.
    assert "GEMINI_API_KEY" not in os.environ


def test_update_privacy_dial_persists_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PUT /settings/privacy-dial round-trips through config_store."""
    import june_brain.config as config_mod
    import june_brain.config_store as store_mod
    from june_brain.routing import UserPrivacyDial

    monkeypatch.setattr(config_mod, "MEMORY_DIR", str(tmp_path))
    monkeypatch.setattr(store_mod, "MEMORY_DIR", str(tmp_path))

    response = settings_route.update_privacy_dial(
        PrivacyDialUpdateRequest(dial="local_only")
    )
    assert response.dial == "local_only"
    assert store_mod.get_privacy_dial() == UserPrivacyDial.LOCAL_ONLY

    response = settings_route.update_privacy_dial(
        PrivacyDialUpdateRequest(dial="cloud_first")
    )
    assert response.dial == "cloud_first"
    assert store_mod.get_privacy_dial() == UserPrivacyDial.CLOUD_FIRST


def test_update_privacy_dial_rejects_unknown_value() -> None:
    """Pydantic validates the Literal at request parse time."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PrivacyDialUpdateRequest(dial="wibble")  # type: ignore[arg-type]
