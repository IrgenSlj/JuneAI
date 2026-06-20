"""Diagnostic runner — orchestrates checks and formats results."""

from __future__ import annotations

import logging
from typing import Any

from .checks import (
    CheckResult,
    check_character,
    check_config,
    check_datadir,
    check_memory_graph,
    check_memory_sqlite,
    check_memory_vector,
    check_provider_generation,
    check_provider_health,
    check_router,
    check_scheduler,
    check_skills_supervisor,
    check_system,
    check_tool_invoke,
    check_tool_registry,
)

logger = logging.getLogger(__name__)

_CHECKS: list[tuple[str, Any]] = [
    ("Config", check_config),
    ("Data directory", check_datadir),
    ("Character", check_character),
    ("Provider health", check_provider_health),
    ("Provider generation", check_provider_generation),
    ("Memory (SQLite)", check_memory_sqlite),
    ("Memory (Vector)", check_memory_vector),
    ("Memory (Graph)", check_memory_graph),
    ("Tool registry", check_tool_registry),
    ("Tool invocation", check_tool_invoke),
    ("Skills supervisor", check_skills_supervisor),
    ("Router", check_router),
    ("Scheduler", check_scheduler),
    ("System", check_system),
]


def run_all() -> list[CheckResult]:
    """Execute every diagnostic check and return the results."""
    results: list[CheckResult] = []
    for name, fn in _CHECKS:
        try:
            result = fn()
        except Exception as exc:
            logger.exception("check %s raised unexpectedly", name)
            result = CheckResult(name=name, status="FAIL", detail=str(exc))
        results.append(result)
    return results


def format_markdown(results: list[CheckResult]) -> str:
    """Render check results as a markdown table."""
    lines = [
        "=== June Diagnostics ===\n",
        f"Ran {len(results)} check(s)\n",
        "",
        "| Check | Status | Detail |",
        "|-------|--------|--------|",
    ]
    passed = 0
    failed = 0
    skipped = 0
    for r in results:
        detail = r.detail.replace("\n", " ").replace("|", "/")
        lines.append(f"| {r.name} | {r.status} | {detail} |")
        if r.status == "PASS":
            passed += 1
        elif r.status == "FAIL":
            failed += 1
        else:
            skipped += 1
    lines.append("")
    total = passed + failed + skipped
    parts: list[str] = []
    parts.append(f"{passed}/{total} passed")
    if failed:
        parts.append(f"{failed} failed")
    if skipped:
        parts.append(f"{skipped} skipped")
    lines.append(", ".join(parts))
    return "\n".join(lines)
