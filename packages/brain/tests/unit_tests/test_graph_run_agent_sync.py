"""Regression tests for synchronous scheduled-agent invocation.

The scheduler invokes the model through the provider stack (no LangChain, no
bound tools). These tests pin that contract with a fake provider.
"""

from __future__ import annotations

from pathlib import Path

from june_brain.providers.base import GenerateResult
from june_brain.providers.registry import ProviderRegistry
from june_brain.scheduler.agent import run_agent_sync


class _FakeProvider:
    model_id = "fake-local"
    tier = "local-fast"

    def __init__(self) -> None:
        self.req = None

    async def generate(self, req):
        self.req = req
        return GenerateResult(
            text="scheduled reply",
            input_tokens=1,
            output_tokens=2,
            latency_ms=3,
            model_id=self.model_id,
            tier="local-fast",
        )

    def stream(self, req):  # pragma: no cover - unused in this path
        raise NotImplementedError

    async def health(self):  # pragma: no cover - unused in this path
        raise NotImplementedError


def test_run_agent_sync_uses_provider_without_bound_tools(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("june_brain.memory.MEMORY_DIR", str(tmp_path))

    fake = _FakeProvider()
    registry = ProviderRegistry(toml_data={"roles": {}, "providers": {}})
    registry.register("local-fast", fake)

    result = run_agent_sync(
        "summarize my day", user_id="scheduled-user", registry=registry
    )

    assert result == "scheduled reply"
    # The user prompt is the last message; no tools are bound.
    assert fake.req.messages[-1].content == "summarize my day"
    assert fake.req.messages[-1].role == "user"
    assert fake.req.messages[0].role == "system"
