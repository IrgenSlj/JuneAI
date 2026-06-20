"""Self-test / diagnostics harness for June.

Each check is a plain function that returns a CheckResult.
The runner in ``runner.py`` runs them all and formats the output.
"""

from __future__ import annotations

from .checks import CheckResult
from .runner import format_markdown, run_all

__all__ = [
    "CheckResult",
    "format_markdown",
    "run_all",
]
