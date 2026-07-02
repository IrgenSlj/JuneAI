"""Run the June API with uvicorn — or a first-party skill's MCP stdio server.

Usage::

    june-api                     # default: boot uvicorn (starts the API + the
                                 # skill supervisor, which spawns skills).
    june-api --run-skill <key>   # run `python -m june_skill_<key>` in-process.

The ``--run-skill`` subcommand exists for the *frozen* PyInstaller sidecar. In
a frozen build ``sys.executable`` is the bootloader (not a Python that
understands ``-m``), and the skill packages ship inside the bundle rather than
on an importable path a subprocess ``python -m`` could reach. The supervisor
therefore re-invokes this same frozen ``june-api`` binary as
``june-api --run-skill <key>``; this branch replicates ``python -m
june_skill_<key>`` by running the module in-process.
"""

from __future__ import annotations

import os
import sys


def _run_skill(key: str) -> None:
    """Run the first-party skill ``june_skill_<key>`` as its MCP stdio server.

    Replicates ``python -m june_skill_<key>`` exactly via :func:`runpy.run_module`
    with ``run_name="__main__"`` so the skill's ``__main__`` block executes.

    Fork-bomb safety: importing ``june_skill_<key>`` imports ``june_brain``,
    which — if it built an agent — would auto-start the skill supervisor and
    spawn the whole skill set again. The parent supervisor already sets
    ``JUNE_IS_SKILL_SUBPROCESS=1`` and ``JUNE_SKILLS_DISABLED=1`` on the child
    env, and ``get_supervisor()`` honors both by skipping auto-start. Defensively,
    if this subcommand is invoked directly (guard not already set), we set both
    here *before* importing the skill module, so the subcommand is safe even
    when run by hand. The normal API path never touches these vars.
    """
    import runpy

    if os.environ.get("JUNE_IS_SKILL_SUBPROCESS") != "1":
        os.environ["JUNE_IS_SKILL_SUBPROCESS"] = "1"
        os.environ["JUNE_SKILLS_DISABLED"] = "1"

    runpy.run_module(f"june_skill_{key}", run_name="__main__")


def main() -> None:
    # Parse argv FIRST — before importing uvicorn or touching the API — so the
    # skill subcommand is a clean, cheap branch that never starts the server.
    argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "--run-skill":
        _run_skill(argv[1])
        return

    import uvicorn

    host = os.getenv("JUNE_API_HOST", "127.0.0.1")
    port = int(os.getenv("JUNE_API_PORT", "8000"))
    reload = os.getenv("JUNE_API_RELOAD", "").lower() in ("1", "true", "yes")

    uvicorn.run("june_api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
