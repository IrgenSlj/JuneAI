"""Module-boundary tripwire for the memory package (S3 decomposition).

``manager.py`` was a 1,160-line god module; S3 split it into focused
modules (paraphrase, recall, writers, extractor) behind a thin facade.
This test fails if the facade starts re-accreting logic, so the seams
that S5/S6/S7 build into stay clean.
"""

from __future__ import annotations

from pathlib import Path

from june_brain.memory import manager

MANAGER_LINE_BUDGET = 250


def test_manager_stays_a_thin_facade() -> None:
    source = Path(manager.__file__)
    line_count = len(source.read_text(encoding="utf-8").splitlines())
    assert line_count < MANAGER_LINE_BUDGET, (
        f"memory/manager.py is {line_count} lines (budget {MANAGER_LINE_BUDGET}). "
        "It is meant to be a thin facade — push new write/recall/extract logic "
        "into writers.py / recall.py / extractor.py / paraphrase.py instead."
    )
