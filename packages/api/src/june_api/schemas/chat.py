"""Chat request and streaming event schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """A single user turn submitted to POST /chat."""

    user_id: str = Field(..., description="Stable identifier for the user. Used as the memory partition key.")
    message: str = Field(..., description="The user's message in plain text.")
    skill: str = Field(
        default="assistant",
        description="Optional skill hint to guide the agent. Defaults to general assistant behavior.",
    )


class ChatEvent(BaseModel):
    """One item in the SSE stream returned by POST /chat.

    The wire format is one ``data:`` frame per event carrying a JSON
    object with this shape. ``type`` drives the UI.
    """

    type: Literal["token", "tool_call", "tool_result", "done", "error"] = Field(
        ..., description="Discriminator that determines the meaning of the payload."
    )
    content: str = Field(
        default="",
        description="Textual content for token and error events; empty otherwise.",
    )
    tool_name: str = Field(
        default="",
        description="Name of the tool for tool_call and tool_result events.",
    )
    tool_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool for tool_call events.",
    )
    tool_result: str = Field(
        default="",
        description="Serialized result returned by the tool for tool_result events.",
    )
