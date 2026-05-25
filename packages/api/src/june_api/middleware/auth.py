"""FastAPI middleware that validates ``X-June-Api-Key`` on every request.

Auth is **opt-in**. June runs locally on the loopback interface as a
single-user app, where an API key adds breakage but no security. The
middleware therefore only enforces a key when ``JUNE_API_AUTH_ENABLED`` is
set (``1``/``true``/``yes``) — intended for deployments that expose the API
beyond ``127.0.0.1``. Off by default.

When enabled, these paths remain exempt (no auth required):
- ``/healthz``
- ``/openapi.json``, ``/docs``, ``/redoc``
- ``/setup/status``, ``/setup/apply``
"""

from __future__ import annotations

import logging
import os

from fastapi import Request, Response
from june_brain.auth import validate_api_key

logger = logging.getLogger(__name__)

_EXEMPT_PREFIXES = (
    "/healthz",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/setup/status",
    "/setup/apply",
)


async def api_key_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Validate ``X-June-Api-Key`` header, or skip for exempt/excluded paths."""

    # Auth is opt-in. Skip entirely unless explicitly enabled.
    if os.getenv("JUNE_API_AUTH_ENABLED", "").strip().lower() not in ("1", "true", "yes"):
        return await call_next(request)

    path = request.url.path
    if any(path.startswith(prefix) for prefix in _EXEMPT_PREFIXES):
        return await call_next(request)

    # OPTIONS preflight — no auth needed
    if request.method == "OPTIONS":
        return await call_next(request)

    provided = request.headers.get("X-June-Api-Key")
    if provided is None:
        logger.warning("Request missing API key: %s %s", request.method, path)
        return Response(
            status_code=401,
            content=b'{"detail":"Missing API key. Provide X-June-Api-Key header."}',
            media_type="application/json",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    if not validate_api_key(provided):
        logger.warning("Request with invalid API key: %s %s", request.method, path)
        return Response(
            status_code=401,
            content=b'{"detail":"Invalid API key."}',
            media_type="application/json",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    return await call_next(request)
