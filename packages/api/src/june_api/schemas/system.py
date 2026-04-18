"""System-status schema (runtime and Ollama)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SystemStatus(BaseModel):
    """What the shells need to display a honest runtime indicator."""

    provider: str = Field(..., description="Active preset key: 'gemma' or 'gemini'.")
    label: str = Field(..., description="Human-readable runtime label.")
    model: str = Field(..., description="Active model identifier.")
    mode: str = Field(..., description="'local' when inference stays on-device, 'api' otherwise.")
    privacy_label: str = Field(..., description="'local-only' or 'api-assisted'.")
    base_url: str = Field(default="", description="Endpoint the brain is talking to.")
    ollama_reachable: bool = Field(
        default=False,
        description="True when the configured Ollama instance responds. Only meaningful for the gemma preset.",
    )
    ollama_has_model: bool = Field(
        default=False,
        description="True when the active Gemma tag is already pulled. Only meaningful for the gemma preset.",
    )
    api_key_present: bool = Field(
        default=False,
        description="True when the active runtime has the credentials it needs. Only meaningful for the gemini preset.",
    )
