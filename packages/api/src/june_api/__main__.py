"""Run the June API with uvicorn."""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.getenv("JUNE_API_HOST", "127.0.0.1")
    port = int(os.getenv("JUNE_API_PORT", "8000"))
    reload = os.getenv("JUNE_API_RELOAD", "").lower() in ("1", "true", "yes")

    uvicorn.run("june_api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
