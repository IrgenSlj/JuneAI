"""Provider boundary types for June's model layer.

All model access goes through a Provider; no raw HTTP model call anywhere else
in the brain. Keeping this independent of LangChain means the boundary is
stable even if the graph internals change.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ToolSpec(BaseModel):
    """A tool advertised to the model for provider-native function calling.

    ``parameters`` is a JSON Schema object (the OpenAI function-calling shape).
    Built from June's ``Tool`` abstraction in the loop layer.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A tool call the model emitted via native function calling."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    messages: list[Message]
    max_tokens: int
    temperature: float = 0.7
    response_format: Literal["text", "json"] = "text"
    stop: list[str] | None = None
    # When set, the provider advertises these tools for native function calling.
    # Providers that lack support ignore them; the loop keeps the prose-JSON
    # extraction path as the fallback (invariant 6).
    tools: list[ToolSpec] | None = None


class GenerateResult(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    model_id: str
    tier: Literal["local-fast", "local-deep", "cloud-capable"]
    # Native tool calls parsed from the response, when the provider/model
    # supports function calling. Empty otherwise.
    tool_calls: list[ToolCall] = Field(default_factory=list)


# JSON-Schema type for June's display-name types (Tool.args carries the latter).
_JSON_TYPES: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def tool_specs_to_openai(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Render ToolSpecs into the OpenAI ``tools`` array shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def parse_openai_tool_calls(message: Any) -> list[ToolCall]:
    """Parse an OpenAI-compatible response message's ``tool_calls`` into ToolCalls.

    Tolerant of missing/garbled arguments JSON — a malformed call is skipped
    rather than raised, so a bad tool emission degrades to no call.
    """
    raw = getattr(message, "tool_calls", None) or []
    calls: list[ToolCall] = []
    for tc in raw:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", "") if fn is not None else ""
        if not name:
            continue
        args_raw = getattr(fn, "arguments", "") or ""
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            args = {}
        calls.append(ToolCall(name=name, arguments=args if isinstance(args, dict) else {}))
    return calls


class ProviderHealth(BaseModel):
    reachable: bool
    loaded: bool = False
    context_window: int | None = None
    detail: str = ""


@runtime_checkable
class Provider(Protocol):
    model_id: str
    tier: str

    async def generate(self, req: GenerateRequest) -> GenerateResult: ...

    def stream(self, req: GenerateRequest) -> AsyncIterator[str]: ...

    async def health(self) -> ProviderHealth: ...
