"""Settings-page schemas.

Distinct from /setup on purpose: /setup is a one-shot flow that lands a user
on chat; /settings is the ongoing control surface for reconfiguring without
blowing away chat history. Neither endpoint ever returns the Gemini key; the
UI only knows whether a key is set and where it lives.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SettingsView(BaseModel):
    """Non-secret snapshot of the active configuration."""

    provider: str = Field(..., description="Active preset key ('gemma' or 'gemini').")
    model: str = Field(..., description="Active model identifier.")
    gemma_model: str = Field(default="", description="Persisted Gemma tag override, if any.")
    gemini_model: str = Field(default="", description="Persisted Gemini model override, if any.")
    ollama_base_url: str = Field(default="", description="Override for the Ollama endpoint, if any.")
    ollama_reachable: bool = Field(default=False, description="Gemma preset only.")
    ollama_has_model: bool = Field(default=False, description="Gemma preset only.")
    api_key_present: bool = Field(default=False, description="Gemini preset only.")
    key_storage: Literal["keyring", "file", "none"] = Field(
        ...,
        description=(
            "Where the Gemini key is stored. 'keyring' means the OS credential store, "
            "'file' means the 0600 config.json fallback, 'none' means no key is set."
        ),
    )
    key_storage_label: str = Field(
        ...,
        description="Human-readable storage label for the settings UI.",
    )


class ForgetKeyResponse(BaseModel):
    """Outcome of POST /settings/forget-key."""

    cleared_from: Literal["keyring", "file", "none"] = Field(
        ...,
        description="Where the key actually lived. 'none' means nothing was stored.",
    )
