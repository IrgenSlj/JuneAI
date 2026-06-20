"""Provider boundary types for June's model layer.

All model access goes through a Provider; no raw HTTP model call anywhere else
in the brain. Keeping this independent of LangChain means the boundary is
stable even if the graph internals change.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class GenerateRequest(BaseModel):
    messages: list[Message]
    max_tokens: int
    temperature: float = 0.7
    response_format: Literal["text", "json"] = "text"
    stop: list[str] | None = None


class GenerateResult(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    model_id: str
    tier: Literal["local-fast", "local-deep", "cloud-capable"]


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
