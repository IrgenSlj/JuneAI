"""FastAPI app factory.

The shells (apps/web, apps/desktop, apps/mobile) all hit this one API.
Routes live in ``june_api.routes``; schemas live in ``june_api.schemas``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from june_brain.activity import ActivityLog
from june_brain.config_store import apply_stored_config_to_env
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .middleware.auth import api_key_middleware
from .schemas import ChatEvent, RecallHit, TaskEventFrame, TaskStepView

logger = logging.getLogger(__name__)

# Paths that produce noise rather than signal — skip them in the activity log.
_ACTIVITY_SKIP_PREFIXES = ("/healthz", "/system/activity", "/openapi.json", "/docs", "/redoc")

apply_stored_config_to_env()


def _cors_origins() -> list[str]:
    """Resolve allowed origins for the browser CORS layer.

    The API binds to ``127.0.0.1`` by default, but that only keeps it off the
    network — it does *not* keep it away from the user's own browser. Any site
    the user visits can issue ``fetch("http://127.0.0.1:8000/...")`; the CORS
    allowlist is what decides whether that site may *read* the response from
    this unauthenticated, memory-bearing API. So the default is an explicit
    allowlist of June's own shells, never ``*``.

    Override with ``JUNE_API_CORS_ORIGINS`` (comma-separated) for deployment
    behind a reverse proxy. Set ``JUNE_API_DEV_OPEN_CORS=1`` to opt into a
    wide-open ``*`` allowlist for local debugging — opt-in and visible, never
    the default.
    """
    raw = os.getenv("JUNE_API_CORS_ORIGINS", "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    if os.getenv("JUNE_API_DEV_OPEN_CORS", "").lower() in ("1", "true", "yes"):
        return ["*"]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "tauri://localhost",
        "capacitor://localhost",
    ]


def _allowed_hosts() -> list[str]:
    """Host-header allowlist — the DNS-rebinding defense for the loopback API.

    A malicious site cannot read our responses cross-origin (CORS handles that),
    but a DNS-rebinding attack can make the browser treat an attacker domain as
    same-origin against 127.0.0.1. Validating the Host header closes that: a
    rebinding request still carries the attacker's domain in Host and is
    rejected here.

    Default covers the loopback names plus ``testserver`` (the in-process
    Starlette TestClient host — unreachable by any browser, so safe to allow).
    Override with ``JUNE_API_ALLOWED_HOSTS`` (comma-separated) when deploying
    behind a real domain; set it to ``*`` to disable host checking.
    """
    raw = os.getenv("JUNE_API_ALLOWED_HOSTS", "").strip()
    if raw:
        return [h.strip() for h in raw.split(",") if h.strip()]
    return ["localhost", "127.0.0.1", "testserver"]


def _maybe_warm_local_model() -> None:
    """Preload the local model so the first chat turn isn't a cold start.

    Best-effort and non-blocking: runs in a daemon thread, only when the
    resolved runtime is local, and skipped under JUNE_SKIP_MODEL_CHECK (tests
    and CI). Never touches the cloud.
    """
    if os.getenv("JUNE_SKIP_MODEL_CHECK", "").lower() in ("1", "true", "yes"):
        return
    try:
        from june_brain import ollama_manager
        from june_brain.config import resolve_runtime_config

        runtime = resolve_runtime_config()
        if not runtime.is_local:
            return
        threading.Thread(
            target=ollama_manager.warm_model,
            args=(runtime.model, runtime.base_url),
            daemon=True,
        ).start()
    except Exception:  # noqa: BLE001
        logger.exception("local model warmup failed to start")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _maybe_warm_local_model()
    yield


def _raise_fd_limit() -> None:
    """Raise the open-file (descriptor) soft limit so long sessions don't hit
    EMFILE ("Too many open files").

    June legitimately holds many descriptors at once: ChromaDB index files,
    per-worker-thread SQLite WAL connections, model/embedding files, and
    keepalive sockets to Ollama. macOS ships a low default soft limit (often
    256), which a busy session can brush against — a burst (streaming + the
    fire-and-forget activity writes + a config save) then tips it over.

    Best-effort and never fatal: no-op on platforms without ``resource``
    (Windows) or when the limit cannot be changed.
    """
    try:
        import resource
    except ImportError:
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = 65536 if hard == resource.RLIM_INFINITY else min(hard, 65536)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError):
        pass


def create_app() -> FastAPI:
    """Assemble the FastAPI instance."""
    _raise_fd_limit()
    from .routes import (
        chat,
        demo,
        greeting,
        memory,
        notifications,
        schedules,
        settings,
        setup,
        skills,
        system,
        tasks,
    )

    app = FastAPI(
        title="June API",
        version="0.2.0",
        description=(
            "HTTP boundary for the June brain. The canonical client is "
            "apps/web (SvelteKit) but any shell speaking JSON/SSE can use it."
        ),
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=_allowed_hosts(),
    )

    @app.middleware("http")
    async def _api_key_check(request: Request, call_next):
        return await api_key_middleware(request, call_next)

    @app.middleware("http")
    async def _record_activity(request: Request, call_next):
        """Log every request to the activity_log table so /system/activity shows it."""
        started = time.monotonic()
        response = await call_next(request)
        latency_ms = int((time.monotonic() - started) * 1000)
        path = request.url.path
        if request.method != "OPTIONS" and not any(
            path.startswith(prefix) for prefix in _ACTIVITY_SKIP_PREFIXES
        ):
            # Offload the SQLite INSERT + prune to a worker thread so the
            # response can return without waiting on a disk write + commit.
            # Fire-and-forget is safe: the activity log is best-effort and
            # any failure is logged inside _record_request_async.
            asyncio.create_task(
                _record_request_async(
                    method=request.method,
                    path=path,
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                )
            )
        return response

    async def _record_request_async(
        *, method: str, path: str, status_code: int, latency_ms: int
    ) -> None:
        try:
            await asyncio.to_thread(
                ActivityLog().record,
                kind="request",
                label=f"{method} {path}",
                status=status_code,
                latency_ms=latency_ms,
            )
        except Exception:  # noqa: BLE001
            logger.exception("activity log write failed")

    app.include_router(chat.router)
    app.include_router(demo.router)
    app.include_router(greeting.router)
    app.include_router(memory.router)
    app.include_router(notifications.router)
    app.include_router(schedules.router)
    app.include_router(settings.router)
    app.include_router(setup.router)
    app.include_router(skills.router)
    app.include_router(system.router)
    app.include_router(tasks.router)

    @app.get("/healthz", tags=["system"])
    def healthz() -> dict[str, str]:
        """Liveness probe — no brain calls, just confirms the process is up."""
        return {"status": "ok"}

    _register_streaming_schemas(app)
    return app


def _register_streaming_schemas(app: FastAPI) -> None:
    """Force ChatEvent (and its nested models) into the OpenAPI components schemas.

    SSE frames aren't a response_model so FastAPI can't discover
    ChatEvent on its own. Without this, the generated TypeScript would
    miss the payload shape for /chat — the most important schema on the
    wire. Models referenced by ChatEvent (e.g. RecallHit) must also be
    registered at the top level so $refs resolve to
    ``#/components/schemas/...`` instead of the per-schema ``$defs``
    Pydantic emits by default.
    """
    base_openapi = app.openapi

    def openapi_with_streaming_schemas() -> dict:
        schema = base_openapi()
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        ref_template = "#/components/schemas/{model}"
        components.setdefault(
            "RecallHit",
            RecallHit.model_json_schema(ref_template=ref_template),
        )
        components.setdefault(
            "ChatEvent",
            ChatEvent.model_json_schema(ref_template=ref_template),
        )
        components.setdefault(
            "TaskStepView",
            TaskStepView.model_json_schema(ref_template=ref_template),
        )
        components.setdefault(
            "TaskEventFrame",
            TaskEventFrame.model_json_schema(ref_template=ref_template),
        )
        return schema

    app.openapi = openapi_with_streaming_schemas  # type: ignore[method-assign]


app = create_app()
