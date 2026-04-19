"""First-run setup schemas.

These drive the /setup route in the web UI. The goal is to turn a fresh
install into a working configuration without requiring the user to touch
the shell or edit files.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SetupStatus(BaseModel):
    """Snapshot used by the UI to decide whether to show /setup or /chat."""

    is_configured: bool = Field(
        ...,
        description=(
            "True once the active provider has everything it needs to run: "
            "Ollama reachable with the Gemma tag pulled, or a Gemini API key present."
        ),
    )
    provider: str = Field(
        default="",
        description="Active preset key ('gemma' or 'gemini'), or empty when no choice has been persisted.",
    )
    model: str = Field(default="", description="Active model identifier, if any.")
    ollama_reachable: bool = Field(default=False, description="Gemma preset only.")
    ollama_has_model: bool = Field(default=False, description="Gemma preset only.")
    api_key_present: bool = Field(default=False, description="Gemini preset only.")


class SetupApplyRequest(BaseModel):
    """User's provider pick from the /setup screen."""

    provider: Literal["gemma", "gemini"] = Field(
        ...,
        description="Which runtime to activate.",
    )
    gemini_api_key: str | None = Field(
        default=None,
        description=(
            "Required when provider is 'gemini'. Stored in config.json with mode 0600 "
            "until native credential storage lands."
        ),
    )
    gemma_model: str | None = Field(
        default=None,
        description="Ollama tag override. Defaults to the preset's default (gemma4:e4b).",
    )
    gemini_model: str | None = Field(
        default=None,
        description="Gemini model override. Defaults to gemini-2.0-flash.",
    )


class SetupApplyResponse(BaseModel):
    """Result of a setup attempt, including a verification round-trip."""

    ok: bool = Field(..., description="True when the chosen provider successfully produced a token.")
    provider: str = Field(..., description="The preset that was applied.")
    model: str = Field(..., description="The model the provider is configured to use.")
    verified: bool = Field(
        ...,
        description="True when a one-shot request to the provider returned a response.",
    )
    message: str = Field(
        default="",
        description="Short, human-readable status. Empty on clean success.",
    )
    hint: str = Field(
        default="",
        description="When ok is false, a specific actionable hint (e.g. 'Ollama not running').",
    )
