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
    embedding_model: str = Field(
        default="",
        description="Configured local embedding model used for semantic recall.",
    )
    embedding_available: bool = Field(
        default=False,
        description="True when the configured local embedding model is available.",
    )
    semantic_recall_status: str = Field(
        default="unknown",
        description="'ready' when semantic recall can embed queries, 'degraded' when it falls back to keyword search.",
    )
    semantic_recall_detail: str = Field(
        default="",
        description="Human-readable explanation of semantic recall readiness.",
    )
    api_key_present: bool = Field(default=False, description="Gemini preset only.")
    user_name: str = Field(default="", description="User's name, if set.")


class SetupApplyRequest(BaseModel):
    """User's provider pick from the /setup screen."""

    provider: Literal["gemma", "gemini"] = Field(
        ...,
        description="Which runtime to activate.",
    )
    user_name: str | None = Field(
        default=None,
        description="User's preferred name. Stored alongside the provider choice.",
    )
    privacy_dial: str | None = Field(
        default=None,
        description="Privacy preference: 'private_by_default', 'cloud_for_smart', etc.",
    )
    gemini_api_key: str | None = Field(
        default=None,
        description=(
            "Required when provider is 'gemini'. Stored in the OS credential "
            "store when available, otherwise in config.json with mode 0600."
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
