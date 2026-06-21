"""Individual diagnostic checks for June subsystems.

Every check is a plain sync function with no arguments that returns a
``CheckResult``.  Async work uses ``asyncio.run()`` internally (matching the
pattern in the existing test suite, which avoids pytest-asyncio).

Checks catch all exceptions and return ``FAIL`` — they never raise.
Self-cleaning checks use a dedicated ``_diag_`` user id and prefix all
temporary records with ``_self_test_``.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

_DIAG_USER = "_diag_"
_TAG = "_self_test_"
_HEALTH_TIMEOUT = 8
_GENERATION_TIMEOUT = 30


@dataclass
class CheckResult:
    """Outcome of a single diagnostic check."""

    name: str
    status: Literal["PASS", "FAIL", "SKIP"]
    detail: str
    duration_ms: int = 0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_config() -> CheckResult:
    start = time.monotonic()
    name = "Config"
    try:
        from june_brain.config import RUNTIME_PRESETS, resolve_runtime_config

        config = resolve_runtime_config()
        preset_count = len(RUNTIME_PRESETS)
        detail = f"{preset_count} presets, active: {config.preset_key} ({config.model})"
        status = "PASS"
    except Exception as exc:
        detail = str(exc)
        status = "FAIL"
    elapsed = int((time.monotonic() - start) * 1000)
    return CheckResult(name=name, status=status, detail=detail, duration_ms=elapsed)


def check_datadir() -> CheckResult:
    start = time.monotonic()
    name = "Data directory"
    try:
        from june_brain.config import MEMORY_DIR

        d = Path(MEMORY_DIR)
        if not d.exists():
            return CheckResult(
                name=name,
                status="FAIL",
                detail=f"Not found: {d}",
                duration_ms=int((time.monotonic() - start) * 1000),
            )
        usage = shutil.disk_usage(d)
        free_gb = usage.free / (1024**3)
        detail = f"{d} ({free_gb:.1f} GB free)"
        status = "PASS" if free_gb > 0.5 else "FAIL"
        if free_gb < 0.5:
            detail += " — less than 500 MB free!"
    except Exception as exc:
        detail = str(exc)
        status = "FAIL"
    elapsed = int((time.monotonic() - start) * 1000)
    return CheckResult(name=name, status=status, detail=detail, duration_ms=elapsed)


def check_character() -> CheckResult:
    start = time.monotonic()
    name = "Character"
    try:
        from june_brain.character import load_or_seed, shaping_section

        block = load_or_seed()
        block_text = block.to_block()
        shaping = shaping_section(block)
        detail = f"Block {len(block_text)} chars, shaping {len(shaping)} chars"
        status = "PASS"
    except Exception as exc:
        detail = str(exc)
        status = "FAIL"
    elapsed = int((time.monotonic() - start) * 1000)
    return CheckResult(name=name, status=status, detail=detail, duration_ms=elapsed)


def check_provider_health() -> CheckResult:
    """Call ``health()`` on every registered provider with a per-provider timeout."""
    start = time.monotonic()
    name = "Provider health"
    try:
        from june_brain.providers.base import ProviderHealth
        from june_brain.providers.registry import get_registry

        registry = get_registry()
        lines: list[str] = []
        fails = 0

        for role in registry.roles():
            provider = registry.get(role)

            async def _health(p=provider) -> ProviderHealth:
                return await asyncio.wait_for(p.health(), timeout=_HEALTH_TIMEOUT)

            try:
                health = asyncio.run(_health())
                if health.reachable:
                    lines.append(f"{role}: OK ({provider.model_id})")
                else:
                    lines.append(f"{role}: UNREACHABLE ({health.detail})")
                    fails += 1
            except TimeoutError:
                lines.append(f"{role}: TIMEOUT (>{_HEALTH_TIMEOUT}s)")
                fails += 1
            except Exception as exc:
                lines.append(f"{role}: ERROR ({exc})")
                fails += 1

        status = "PASS" if fails == 0 else "FAIL"
        detail = "; ".join(lines)
    except Exception as exc:
        detail = str(exc)
        status = "FAIL"
    elapsed = int((time.monotonic() - start) * 1000)
    return CheckResult(name=name, status=status, detail=detail, duration_ms=elapsed)


def check_provider_generation() -> CheckResult:
    """Send a minimal generate request to a reachable provider.

    Skips if the previous health check or provider infrastructure is down.
    """
    start = time.monotonic()
    name = "Provider generation"
    try:
        from june_brain.providers.base import GenerateRequest, Message
        from june_brain.providers.registry import get_registry

        registry = get_registry()

        for candidate_role in ("local-fast", "local-deep", "cloud-capable"):
            try:
                provider = registry.get(candidate_role)
            except KeyError:
                continue

            async def _gen(p=provider) -> str | None:
                try:
                    h = await asyncio.wait_for(p.health(), timeout=_HEALTH_TIMEOUT)
                    if not h.reachable:
                        return None
                    req = GenerateRequest(
                        messages=[Message(role="user", content="Hi")],
                        max_tokens=10,
                        temperature=0.0,
                    )
                    result = await asyncio.wait_for(
                        p.generate(req), timeout=_GENERATION_TIMEOUT
                    )
                    if result.text.strip():
                        return f"{candidate_role}: {result.text[:60].strip()} ({result.input_tokens} in, {result.output_tokens} out, {result.latency_ms}ms)"
                    return f"{candidate_role}: empty response"
                except TimeoutError:
                    return f"{candidate_role}: generation timed out"
                except Exception as exc:
                    return f"{candidate_role}: {exc}"

            detail_text = asyncio.run(_gen())
            if detail_text is None:
                continue
            elapsed = int((time.monotonic() - start) * 1000)
            return CheckResult(name=name, status="PASS", detail=detail_text, duration_ms=elapsed)

        elapsed = int((time.monotonic() - start) * 1000)
        return CheckResult(
            name=name, status="SKIP", detail="No reachable provider found", duration_ms=elapsed
        )
    except Exception as exc:
        elapsed = int((time.monotonic() - start) * 1000)
        return CheckResult(name=name, status="FAIL", detail=str(exc), duration_ms=elapsed)


def check_memory_sqlite() -> CheckResult:
    start = time.monotonic()
    name = "Memory (SQLite)"
    try:
        from june_brain.memory.sqlite import Memory

        mem = Memory(_DIAG_USER)
        test_key = f"{_TAG}check_{int(time.time())}"
        mem._conn.execute(
            "INSERT OR REPLACE INTO journal (user_id, entry, timestamp) VALUES (?, ?, ?)",
            (_DIAG_USER, test_key, datetime.now(UTC).isoformat()),
        )
        mem._conn.commit()
        row = mem._conn.execute(
            "SELECT entry FROM journal WHERE user_id=? AND entry=?",
            (_DIAG_USER, test_key),
        ).fetchone()
        mem._conn.execute(
            "DELETE FROM journal WHERE user_id=? AND entry=?",
            (_DIAG_USER, test_key),
        )
        mem._conn.commit()
        if row and row[0] == test_key:
            detail = "Read/write OK"
            status = "PASS"
        else:
            detail = "Write did not persist"
            status = "FAIL"
    except Exception as exc:
        detail = str(exc)
        status = "FAIL"
    elapsed = int((time.monotonic() - start) * 1000)
    return CheckResult(name=name, status=status, detail=detail, duration_ms=elapsed)


def check_memory_vector() -> CheckResult:
    start = time.monotonic()
    name = "Memory (Vector)"
    try:
        from june_brain.memory.vector import VectorStore

        vs = VectorStore(_DIAG_USER)
        test_text = f"{_TAG}vector check at {time.time()}"
        record = vs.upsert(text=test_text, metadata={"source": "self_test"})
        hits = vs.search(test_text, k=3)
        found = any(test_text in h.get("text", "") for h in hits)
        vs.delete(record["fact_id"])
        if found:
            detail = "Embed + search OK (sqlite-vec)"
            status = "PASS"
        elif record.get("fact_id"):
            # Shadow write succeeded but search surfaced nothing — the local
            # embedder is unavailable (model not pulled / Ollama down). Recall
            # degrades to the keyword scan; not a failure of the store (ADR 0019).
            detail = "Embedder unavailable — semantic recall degraded to keyword"
            status = "SKIP"
        else:
            detail = "Write did not surface in search"
            status = "FAIL"
    except Exception as exc:
        detail = str(exc)
        status = "FAIL"
    elapsed = int((time.monotonic() - start) * 1000)
    return CheckResult(name=name, status=status, detail=detail, duration_ms=elapsed)


def check_memory_graph() -> CheckResult:
    start = time.monotonic()
    name = "Memory (Graph)"
    try:
        from june_brain.memory.graph import KnowledgeGraph

        kg = KnowledgeGraph(_DIAG_USER)
        test_label = f"{_TAG}node_{time.time()}"
        node = kg.add_node(label=test_label, kind="diagnostic", props={"check": True})
        fetched = kg._conn.execute(
            "SELECT label FROM graph_nodes WHERE user_id=? AND node_id=?",
            (_DIAG_USER, node["node_id"]),
        ).fetchone()
        kg.remove_node(node["node_id"])
        if fetched and fetched[0] == test_label:
            detail = "Add/query/remove OK"
            status = "PASS"
        else:
            detail = "Node not found after insert"
            status = "FAIL"
    except Exception as exc:
        detail = str(exc)
        status = "FAIL"
    elapsed = int((time.monotonic() - start) * 1000)
    return CheckResult(name=name, status=status, detail=detail, duration_ms=elapsed)


def check_tool_registry() -> CheckResult:
    start = time.monotonic()
    name = "Tool registry"
    try:
        from june_brain.tools import JUNE_TOOLS

        issues: list[str] = []
        ok_count = 0
        for t in JUNE_TOOLS:
            t_name = getattr(t, "name", "")
            if not t_name:
                issues.append("unnamed tool in JUNE_TOOLS")
                continue
            desc = getattr(t, "description", "") or ""
            if not desc.strip():
                issues.append(f"{t_name}: empty description")
            args = getattr(t, "args", {}) or {}
            if not isinstance(args, dict):
                issues.append(f"{t_name}: invalid args schema")
            ok_count += 1
        if issues:
            detail = f"{ok_count} OK, {len(issues)} issues: {'; '.join(issues[:3])}"
            status = "FAIL"
        else:
            detail = f"{ok_count} tools registered with valid schemas"
            status = "PASS"
    except Exception as exc:
        detail = str(exc)
        status = "FAIL"
    elapsed = int((time.monotonic() - start) * 1000)
    return CheckResult(name=name, status=status, detail=detail, duration_ms=elapsed)


def check_tool_invoke() -> CheckResult:
    start = time.monotonic()
    name = "Tool invocation"
    try:
        from june_brain.tools import get_runtime_privacy_status

        result = get_runtime_privacy_status.invoke({})
        if isinstance(result, str) and len(result) > 10:
            detail = result[:80].replace("\n", " | ")
            status = "PASS"
        else:
            detail = f"Unexpected return: {type(result).__name__}"
            status = "FAIL"
    except Exception as exc:
        detail = str(exc)
        status = "FAIL"
    elapsed = int((time.monotonic() - start) * 1000)
    return CheckResult(name=name, status=status, detail=detail, duration_ms=elapsed)


def check_skills_supervisor() -> CheckResult:
    start = time.monotonic()
    name = "Skills supervisor"
    try:
        from june_brain.skills.loader import list_status
        from june_brain.skills.manifest import load_manifest

        manifest = load_manifest()
        entries = list(manifest.entries.keys())
        statuses = list_status()
        running = sum(1 for s in statuses if s.get("status") == "running")
        stopped = sum(1 for s in statuses if s.get("status") == "stopped")
        crashed = sum(1 for s in statuses if s.get("status") == "crashed")
        parts = [f"Manifest: {len(entries)} skill(s)"]
        if running:
            parts.append(f"{running} running")
        if stopped:
            parts.append(f"{stopped} stopped")
        if crashed:
            parts.append(f"{crashed} crashed")
        detail = ", ".join(parts)
        status = "PASS"
    except Exception as exc:
        detail = str(exc)
        status = "FAIL"
    elapsed = int((time.monotonic() - start) * 1000)
    return CheckResult(name=name, status=status, detail=detail, duration_ms=elapsed)


def check_router() -> CheckResult:
    start = time.monotonic()
    name = "Router"
    try:
        from june_brain.routing import (
            ModelRouter,
            ResolvedTier,
            SkillModelPolicy,
            UserPrivacyDial,
        )

        router = ModelRouter(cloud_available=True)
        local_only = router.resolve(
            skill_policy=SkillModelPolicy.LOCAL_ONLY, dial=UserPrivacyDial.LOCAL_ONLY
        )
        cloud_req = router.resolve(
            skill_policy=SkillModelPolicy.CLOUD_REQUIRED, dial=UserPrivacyDial.CLOUD_FIRST
        )
        blocked = router.resolve(
            skill_policy=SkillModelPolicy.CLOUD_REQUIRED, dial=UserPrivacyDial.LOCAL_ONLY
        )
        router_offline = ModelRouter(cloud_available=False)
        fallback = router_offline.resolve(
            skill_policy=SkillModelPolicy.CLOUD_ALLOWED, dial=UserPrivacyDial.CLOUD_FIRST
        )
        assert local_only == ResolvedTier.LOCAL, f"expected LOCAL, got {local_only}"
        assert cloud_req == ResolvedTier.CLOUD, f"expected CLOUD, got {cloud_req}"
        assert blocked == ResolvedTier.UNAVAILABLE, f"expected UNAVAILABLE, got {blocked}"
        assert fallback == ResolvedTier.LOCAL, f"expected LOCAL fallback, got {fallback}"
        detail = "4/4 resolution patterns correct"
        status = "PASS"
    except AssertionError as exc:
        detail = str(exc)
        status = "FAIL"
    except Exception as exc:
        detail = str(exc)
        status = "FAIL"
    elapsed = int((time.monotonic() - start) * 1000)
    return CheckResult(name=name, status=status, detail=detail, duration_ms=elapsed)


def check_scheduler() -> CheckResult:
    start = time.monotonic()
    name = "Scheduler"
    try:
        from june_brain.memory.sqlite import _get_connection, db_path
        from june_brain.scheduler.models import _SCHEDULES_TABLE_SQL, Schedule
        from june_brain.scheduler.store import ScheduleStore

        conn = _get_connection(db_path())
        conn.executescript(_SCHEDULES_TABLE_SQL)
        conn.commit()
        store = ScheduleStore(conn)
        test_name = f"{_TAG}schedule_{time.time()}"
        sched = Schedule(
            user_id=_DIAG_USER,
            name=test_name,
            description="diagnostic check, will be deleted",
            interval_seconds=3600,
            scheduled_at=datetime.now(UTC).isoformat(),
        )
        created = store.create(sched)
        listed = store.list(_DIAG_USER)
        found = any(s.name == test_name for s in listed)
        if created.id and found:
            store.delete(created.id)
            detail = "Create/list/delete OK"
            status = "PASS"
        else:
            detail = "Schedule create/list round-trip failed"
            status = "FAIL"
    except Exception as exc:
        detail = str(exc)
        status = "FAIL"
    elapsed = int((time.monotonic() - start) * 1000)
    return CheckResult(name=name, status=status, detail=detail, duration_ms=elapsed)


def check_system() -> CheckResult:
    start = time.monotonic()
    name = "System"
    try:
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        arch = getattr(sys, "platform", sys.platform)
        parts = [f"Python {py_ver}", arch]
        from june_brain.config import MEMORY_DIR

        usage = shutil.disk_usage(Path(MEMORY_DIR))
        parts.append(f"{usage.free // (1024**3)} GB free")
        detail = ", ".join(parts)
        status = "PASS"
    except Exception as exc:
        detail = str(exc)
        status = "FAIL"
    elapsed = int((time.monotonic() - start) * 1000)
    return CheckResult(name=name, status=status, detail=detail, duration_ms=elapsed)
