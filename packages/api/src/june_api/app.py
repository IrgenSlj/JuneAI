"""FastAPI app factory.

The shells (apps/web, apps/desktop, apps/mobile) all hit this one API.
Routes live in ``june_api.routes``; schemas live in ``june_api.schemas``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from june_brain.activity import ActivityLog
from june_brain.config_store import apply_stored_config_to_env

from .schemas import ChatEvent, RecallHit

logger = logging.getLogger(__name__)

# Paths that produce noise rather than signal — skip them in the activity log.
_ACTIVITY_SKIP_PREFIXES = ("/healthz", "/system/activity", "/openapi.json", "/docs", "/redoc")

apply_stored_config_to_env()


def _cors_origins() -> list[str]:
    """Resolve allowed origins from env.

    Defaults cover the SvelteKit dev server (apps/web) plus the Tauri
    desktop shell and Capacitor mobile shell which load the built app
    from custom schemes. Override with ``JUNE_API_CORS_ORIGINS``
    (comma-separated) in production.
    """
    raw = os.getenv("JUNE_API_CORS_ORIGINS", "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
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


def create_app() -> FastAPI:
    """Assemble the FastAPI instance."""
    from .routes import chat, demo, memory, obsidian, settings, setup, skills, system, tasks

    app = FastAPI(
        title="June API",
        version="0.2.0",
        description=(
            "HTTP boundary for the June brain. The canonical client is "
            "apps/web (SvelteKit) but any shell speaking JSON/SSE can use it."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

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
    app.include_router(memory.router)
    app.include_router(obsidian.router)
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
        return schema

    app.openapi = openapi_with_streaming_schemas  # type: ignore[method-assign]


app = create_app()
