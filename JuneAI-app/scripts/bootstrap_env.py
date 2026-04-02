#!/usr/bin/env python3
"""Bootstrap the JuneAI development environment."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
if sys.platform == "win32":
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"


def run(cmd: list[str]) -> None:
    """Run a command from the project root."""
    subprocess.run(cmd, cwd=ROOT, check=True)


def ensure_python_version() -> None:
    """Reject interpreters older than the project minimum."""
    if sys.version_info < (3, 9):
        raise SystemExit("JuneAI requires Python 3.9 or newer.")


def create_venv(force: bool) -> None:
    """Create the virtual environment if needed."""
    if force and VENV_DIR.exists():
        shutil.rmtree(VENV_DIR)
    if not VENV_DIR.exists():
        run([sys.executable, "-m", "venv", str(VENV_DIR)])


def install_dependencies() -> None:
    """Install editable project dependencies into the venv."""
    run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(VENV_PYTHON), "-m", "pip", "install", "-e", ".[dev]"])


def verify_install() -> None:
    """Run the environment verification script."""
    run([str(VENV_PYTHON), str(ROOT / "scripts" / "verify_env.py")])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse bootstrap flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recreate .venv before installing.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip the post-install verification step.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Bootstrap the local development environment."""
    args = parse_args(argv)
    ensure_python_version()
    create_venv(force=args.force)
    install_dependencies()
    if not args.skip_verify:
        verify_install()
    run_command = f"{VENV_PYTHON} -m streamlit run app.py --server.headless true --server.port 8501"
    print("Bootstrap complete.")
    print(f"Run: `make run` or `{run_command}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
