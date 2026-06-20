"""Synchronous, non-streaming model invocation for scheduled jobs.

Scheduled jobs are user-requested and deterministic (briefings, reminders);
June never creates them on a timer for proactive engagement (ADR 0016). They
are informational, so no tools are bound — the invocation just builds a system
prompt plus the user ``prompt`` and returns the text reply.

This uses the provider stack (``providers.registry``) directly, keeping the
scheduler free of LangChain. Local-fast is the default role: a background job
should not silently reach the network (invariants 1 and 2).
"""

from __future__ import annotations

import asyncio

from ..providers.base import GenerateRequest, Message
from ..providers.registry import ProviderRegistry, get_registry


def run_agent_sync(
    prompt: str,
    user_id: str,
    *,
    registry: ProviderRegistry | None = None,
    role: str = "local-fast",
) -> str:
    """Build a minimal system prompt + ``prompt`` and return the model's text.

    No tools are bound — scheduled invocations are informational and must not
    trigger side effects.
    """
    from ..memory import Memory
    from ..skills.prompts import build_system_prompt

    memory = Memory(user_id)
    system = build_system_prompt("default", memory=memory)

    reg = registry if registry is not None else get_registry()
    provider = reg.get(role)

    req = GenerateRequest(
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=prompt),
        ],
        max_tokens=1024,
    )
    result = asyncio.run(provider.generate(req))
    return result.text
