"""June's intelligence layer: agent, memory, runtime, skills, patterns."""

from typing import Any

from .graph import create_june_agent, get_or_create_agent, invalidate_agent


def __getattr__(name: str) -> Any:
    if name == "june_agent":
        from . import graph

        return graph.june_agent
    raise AttributeError(name)

__all__ = ["create_june_agent", "get_or_create_agent", "invalidate_agent", "june_agent"]
