"""Tests for the self-test / diagnostics harness.

Covers the deterministic, network-free checks plus the markdown formatter.
The provider health/generation checks hit live providers, so they are not
exercised here — only the pure checks and the runner's formatting contract.
"""

from __future__ import annotations

from june_brain.self_test import CheckResult, format_markdown
from june_brain.self_test.checks import (
    check_config,
    check_router,
    check_system,
    check_tool_registry,
)


def test_check_config_passes():
    result = check_config()
    assert result.status == "PASS", result.detail
    assert "presets" in result.detail


def test_check_router_resolution_patterns():
    result = check_router()
    assert result.status == "PASS", result.detail
    assert "4/4" in result.detail


def test_check_tool_registry_all_valid():
    result = check_tool_registry()
    assert result.status == "PASS", result.detail
    assert "valid schemas" in result.detail


def test_check_system_reports_python_version():
    result = check_system()
    assert result.status == "PASS", result.detail
    assert "Python" in result.detail


def test_format_markdown_counts_statuses():
    results = [
        CheckResult(name="A", status="PASS", detail="ok"),
        CheckResult(name="B", status="FAIL", detail="boom"),
        CheckResult(name="C", status="SKIP", detail="n/a"),
    ]
    rendered = format_markdown(results)
    assert "3 check(s)" in rendered
    assert "1/3 passed" in rendered
    assert "1 failed" in rendered
    assert "1 skipped" in rendered


def test_format_markdown_escapes_pipes_and_newlines():
    """Detail text must not break the markdown table layout."""
    results = [CheckResult(name="X", status="FAIL", detail="line1\nwith | pipe")]
    rendered = format_markdown(results)
    body = [ln for ln in rendered.splitlines() if ln.startswith("| X ")][0]
    assert "\n" not in body[2:]
    # The original pipe in the detail is replaced so it can't split the cell.
    assert "with / pipe" in body
