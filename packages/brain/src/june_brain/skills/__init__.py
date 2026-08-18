"""June's skills layer.

Two related concepts share this namespace:

1. **Conversational personas** (assistant/planner/wellness/curator) —
   the skill the chat adopts for a given user query. Lives in
   :mod:`.prompts` and is re-exported here for backward compatibility.

2. **MCP skill servers** (calendar/research/files/telegram) —
   external processes that expose tools via the Model Context Protocol.
   Lives in :mod:`.loader`, :mod:`.supervisor`, and :mod:`.manifest`.

Importing this module does not spawn any MCP processes — callers must
call :func:`load_skill_tools` or start a :class:`SkillSupervisor`
explicitly.
"""

from .loader import (
    SkillManifest,
    SkillManifestEntry,
    SkillProcess,
    SkillStatus,
    SkillSupervisor,
    available_tools,
    get_supervisor,
    list_status,
    load_manifest,
    load_skill_tools,
    manifest_path,
    reload_skills,
    reset_supervisor_for_tests,
    save_manifest,
    set_skill_enabled,
)
from .manifest import DEFAULT_MANIFEST
from .prompts import (
    DEFAULT_SKILL,
    SKILLS,
    SkillDefinition,
    build_system_prompt,
    infer_skill_from_text,
)

__all__ = [
    # Personas
    "DEFAULT_SKILL",
    "SKILLS",
    "SkillDefinition",
    "build_system_prompt",
    "infer_skill_from_text",
    # MCP skills
    "DEFAULT_MANIFEST",
    "SkillManifest",
    "SkillManifestEntry",
    "SkillProcess",
    "SkillStatus",
    "SkillSupervisor",
    "available_tools",
    "get_supervisor",
    "list_status",
    "load_manifest",
    "load_skill_tools",
    "manifest_path",
    "reload_skills",
    "reset_supervisor_for_tests",
    "save_manifest",
    "set_skill_enabled",
]
