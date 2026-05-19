"""FastAPI app factory.

The shells (apps/web, apps/desktop, apps/mobile) all hit this one API.
Routes live in ``june_api.routes``; schemas live in ``june_api.schemas``.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from june_brain.config_store import apply_stored_config_to_env

from .schemas import ChatEvent, RecallHit

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
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

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
