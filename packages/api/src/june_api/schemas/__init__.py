"""Pydantic schemas for the June API.

These define the request/response shape on the wire and feed the
OpenAPI spec that tools/codegen.sh converts to TypeScript for the UI.
"""

from .chat import ChatEvent, ChatRequest
from .memory import MemoryDeleteResponse, MemoryFact, MemorySnapshot
from .setup import SetupApplyRequest, SetupApplyResponse, SetupStatus
from .skills import (
    SkillInfo,
    SkillsResponse,
    SkillToggleRequest,
    SkillToggleResponse,
    SkillToolInfo,
)
from .system import SystemStatus

__all__ = [
    "ChatEvent",
    "ChatRequest",
    "MemoryDeleteResponse",
    "MemoryFact",
    "MemorySnapshot",
    "SetupApplyRequest",
    "SetupApplyResponse",
    "SetupStatus",
    "SkillInfo",
    "SkillsResponse",
    "SkillToggleRequest",
    "SkillToggleResponse",
    "SkillToolInfo",
    "SystemStatus",
]
