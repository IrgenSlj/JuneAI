"""Tests for the june-api entry point argv dispatch (`june_api.__main__`).

Two branches:
  - `june-api --run-skill <key>`  -> runpy-launch june_skill_<key> (the frozen
    sidecar path), never starting uvicorn.
  - `june-api`                    -> uvicorn.run(...) (the default API path).

Both `runpy.run_module` and `uvicorn.run` are patched so nothing actually
launches. We also assert the fork-bomb guard env vars are set in the skill branch.
"""

from __future__ import annotations

import os
import runpy
import sys

import uvicorn
from june_api.__main__ import main


def test_run_skill_dispatches_to_run_module(monkeypatch):
    calls: dict[str, object] = {}

    def fake_run_module(mod_name, run_name=None):
        calls["mod_name"] = mod_name
        calls["run_name"] = run_name

    def fail_uvicorn(*args, **kwargs):
        raise AssertionError("uvicorn.run must not be called for --run-skill")

    monkeypatch.setattr(runpy, "run_module", fake_run_module)
    monkeypatch.setattr(uvicorn, "run", fail_uvicorn)
    monkeypatch.setattr(sys, "argv", ["june-api", "--run-skill", "calendar"])

    # Guard not set yet -> the branch must set both defensively before importing
    # the skill module. delenv records absence so monkeypatch removes the keys
    # our code sets directly on teardown.
    monkeypatch.delenv("JUNE_IS_SKILL_SUBPROCESS", raising=False)
    monkeypatch.delenv("JUNE_SKILLS_DISABLED", raising=False)

    main()

    assert calls == {"mod_name": "june_skill_calendar", "run_name": "__main__"}
    assert os.environ["JUNE_IS_SKILL_SUBPROCESS"] == "1"
    assert os.environ["JUNE_SKILLS_DISABLED"] == "1"


def test_run_skill_preserves_existing_guard(monkeypatch):
    # When the parent supervisor already set the guard, the branch must not
    # clobber it (and stays fork-bomb safe either way).
    monkeypatch.setattr(runpy, "run_module", lambda *a, **k: None)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["june-api", "--run-skill", "health"])
    monkeypatch.setenv("JUNE_IS_SKILL_SUBPROCESS", "1")
    monkeypatch.delenv("JUNE_SKILLS_DISABLED", raising=False)

    main()

    # Already-guarded: we don't force JUNE_SKILLS_DISABLED on (the supervisor
    # sets both together; the existing guard already prevents re-spawn).
    assert os.environ["JUNE_IS_SKILL_SUBPROCESS"] == "1"
    assert "JUNE_SKILLS_DISABLED" not in os.environ


def test_default_argv_calls_uvicorn(monkeypatch):
    called: dict[str, object] = {}

    def fake_uvicorn_run(app, **kwargs):
        called["app"] = app
        called["kwargs"] = kwargs

    def fail_run_module(*args, **kwargs):
        raise AssertionError("runpy.run_module must not be called for the default path")

    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)
    monkeypatch.setattr(runpy, "run_module", fail_run_module)
    monkeypatch.setattr(sys, "argv", ["june-api"])

    main()

    assert called["app"] == "june_api.app:app"
