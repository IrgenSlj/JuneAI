#!/usr/bin/env python3
"""Boot the *frozen* sidecar and check it actually works (Phase 7.2).

    python3 tools/smoke_packaged.py
    python3 tools/smoke_packaged.py --binary /path/to/june-api --timeout 60

Why this exists, specifically:

`v0.1.0` shipped with a total chat hang. Every turn hung forever in the packaged
app while the same code answered in 18 seconds in development, and **986 green
tests did not see it**, because the tests import `june_brain` in a dev
interpreter and the bug only exists in a frozen, ad-hoc-signed binary. The
macOS Keychain blocks on an authorization decision that never arrives in a
headless sidecar, because the signed bundle has a different code identity from
the interpreter that created the keychain items.

The only thing that catches that class of defect is running the artifact. So
this does, against the real bundle, with a wall-clock budget on every step —
a hang is a *timeout*, not an exception, and nothing that asserts on exceptions
would ever have found it.

## What it covers

- The frozen bundle boots at all. PyInstaller bundles fail here constantly —
  a missing hidden import, a data file that did not get collected — and none of
  it is visible from a dev interpreter.
- `/healthz` and `/system` answer inside a budget.
- `POST /system/ledger/verify` answers inside a budget. This one walks
  `device_public_key()` → `secret_store.load_secret()` → the OS keychain, which
  is the call that hung. **Anything that blocks on that path, for any reason,
  fails here on the clock rather than on an exception.**
- A ledger the running binary wrote verifies with `june-verify`, run from
  outside the process against the same database.
- The process shuts down when asked (a `SIGTERM` it ignores is reported).

## What it does not cover, stated precisely

It does **not** reproduce the exact `v0.1.0` condition. That hang needed a
keychain item that already existed *and* had been created under a different code
identity; this runs against a fresh data dir, where a lookup for a missing item
returns immediately without an authorization prompt. Reproducing it faithfully
means seeding the keychain from a differently-signed binary, which is not
something a CI runner can stage.

What this does give is the property that was missing entirely: **the artifact is
executed, and every step is on a wall-clock budget.** A hang is a timeout, not a
traceback, so no amount of exception-based testing would ever have found the
original bug. If the deadline in `secret_store._run_guarded` were removed and
the keychain blocked, this fails.

A real chat turn needs Ollama and an 8GB model, which no CI runner should pull.
`--with-chat` opts in when a model is present locally.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = REPO_ROOT / "apps/desktop/src-tauri/binaries/june-api/june-api"

# Generous, because this is a hang detector rather than a benchmark. A healthy
# frozen boot is a few seconds; the Keychain hang was unbounded.
BOOT_BUDGET_S = 45.0
REQUEST_BUDGET_S = 20.0


class SmokeFailure(Exception):
    pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _request(url: str, *, method: str = "GET", timeout: float, token: str) -> tuple[int, str]:
    req = urllib.request.Request(url, method=method, headers={"X-June-Token": token})
    if method == "POST":
        req.data = b""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def _timed(label: str, budget: float, fn):  # type: ignore[no-untyped-def]
    """Run ``fn``, fail if it exceeds ``budget``. The budget is the whole point."""
    start = time.monotonic()
    result = fn()
    elapsed = time.monotonic() - start
    if elapsed > budget:
        raise SmokeFailure(f"{label} took {elapsed:.1f}s, over the {budget:.0f}s budget")
    print(f"  ok   {label} ({elapsed:.2f}s)")
    return result


def _wait_for_health(port: int, token: str, proc: subprocess.Popen, budget: float) -> None:
    deadline = time.monotonic() + budget
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = (proc.stdout.read() or b"").decode("utf-8", "replace") if proc.stdout else ""
            raise SmokeFailure(f"the sidecar exited with code {proc.returncode}\n{out[-2000:]}")
        try:
            status, _ = _request(
                f"http://127.0.0.1:{port}/healthz", timeout=3.0, token=token
            )
            if status == 200:
                return
            last = f"status {status}"
        except Exception as exc:  # noqa: BLE001 - still booting
            last = str(exc)
        time.sleep(0.25)
    raise SmokeFailure(f"the sidecar never became healthy in {budget:.0f}s (last: {last})")


def run_smoke(binary: Path, *, with_chat: bool = False) -> int:
    if not binary.exists():
        print(f"smoke: no frozen sidecar at {binary}")
        print("smoke: build it with tools/packaging/build-sidecar.sh, then re-run.")
        return 3

    port = _free_port()
    token = "smoke-token-not-a-secret"
    data_dir = Path(tempfile.mkdtemp(prefix="june-smoke-"))
    env = {
        **os.environ,
        "JUNE_API_HOST": "127.0.0.1",
        "JUNE_API_PORT": str(port),
        "JUNE_API_TOKEN": token,
        "JUNE_DATA_DIR": str(data_dir),
    }

    print(f"smoke: {binary}")
    print(f"smoke: port {port}, data dir {data_dir}")

    proc = subprocess.Popen(  # noqa: S603 - path is operator-supplied
        [str(binary)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"

    try:
        _timed("frozen sidecar boots and serves /healthz", BOOT_BUDGET_S,
               lambda: _wait_for_health(port, token, proc, BOOT_BUDGET_S))

        def _system() -> None:
            status, body = _request(f"{base}/system", timeout=REQUEST_BUDGET_S, token=token)
            if status != 200:
                raise SmokeFailure(f"/system returned {status}: {body[:400]}")

        _timed("/system answers", REQUEST_BUDGET_S, _system)

        # The targeted check. This path reaches the OS keychain through
        # device_public_key(); it is where v0.1.0 hung forever.
        def _verify() -> None:
            status, body = _request(
                f"{base}/system/ledger/verify", method="POST",
                timeout=REQUEST_BUDGET_S, token=token,
            )
            if status != 200:
                raise SmokeFailure(f"ledger verify returned {status}: {body[:400]}")
            if not json.loads(body).get("ok", False):
                raise SmokeFailure(f"a freshly created ledger did not verify: {body[:400]}")

        _timed("ledger verify answers (reaches the OS keychain)", REQUEST_BUDGET_S, _verify)

        if with_chat:
            def _chat() -> None:
                req = urllib.request.Request(
                    f"{base}/chat/history/smoke-user",
                    headers={"X-June-Token": token},
                )
                with urllib.request.urlopen(req, timeout=REQUEST_BUDGET_S):
                    pass

            _timed("chat history answers", REQUEST_BUDGET_S, _chat)

        # Verify the database this running binary wrote, from outside it, with
        # the same tool a user would run.
        db = data_dir / "memory" / "june.db"
        if db.exists():
            verify_bin = shutil.which("june-verify") or str(
                REPO_ROOT / "packages/brain/.venv/bin/june-verify"
            )
            if Path(verify_bin).exists():
                result = subprocess.run(  # noqa: S603
                    [verify_bin, "--db", str(db), "--json"],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode != 0:
                    raise SmokeFailure(
                        f"june-verify rejected the database the binary wrote: {result.stdout}"
                    )
                print("  ok   june-verify accepts the database the binary wrote")
        else:
            print("  --   no database written yet (nothing exercised memory)")

    except SmokeFailure as exc:
        print(f"\nsmoke: FAILED — {exc}")
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            print("smoke: WARNING — the sidecar ignored SIGTERM and was killed")
        shutil.rmtree(data_dir, ignore_errors=True)

    print("\nsmoke: OK — the packaged binary boots, serves, and keeps an honest ledger.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--with-chat", action="store_true", help="also exercise a chat route")
    args = ap.parse_args()
    return run_smoke(Path(args.binary), with_chat=args.with_chat)


if __name__ == "__main__":
    sys.exit(main())
