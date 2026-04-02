#!/usr/bin/env python3
"""Start the JuneAI app and verify that Streamlit serves a 200 response."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def wait_for_http_ok(url: str, timeout: float, proc: subprocess.Popen[bytes]) -> None:
    """Wait for the Streamlit app to return HTTP 200."""
    deadline = time.monotonic() + timeout
    last_error: str | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"Streamlit exited early with code {proc.returncode}.")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise SystemExit(f"Timed out waiting for {url} ({last_error or 'no response'}).")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse smoke-test options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501, help="Port to probe.")
    parser.add_argument("--timeout", type=float, default=45.0, help="Seconds to wait for HTTP 200.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Start Streamlit, wait for readiness, and shut it down cleanly."""
    args = parse_args(argv)
    url = f"http://127.0.0.1:{args.port}"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless",
        "true",
        "--server.port",
        str(args.port),
    ]
    proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        wait_for_http_ok(url, args.timeout, proc)
        print(f"Smoke test passed: {url}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
