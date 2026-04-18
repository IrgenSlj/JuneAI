"""Pydantic schemas for the June API.

These define the request/response shape on the wire and feed the
OpenAPI spec that tools/codegen.sh converts to TypeScript for the UI.
"""

from .chat import ChatEvent, ChatRequest
from .memory import MemoryFact, MemorySnapshot
from .skills import SkillInfo, SkillsResponse
from .system import SystemStatus

__all__ = [
    "ChatEvent",
    "ChatRequest",
    "MemoryFact",
    "MemorySnapshot",
    "SkillInfo",
    "SkillsResponse",
    "SystemStatus",
]
