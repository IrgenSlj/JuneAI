"""Unit tests for the reliability harness math (S5.3)."""

from __future__ import annotations

import pytest
from june_brain.experiments.reliability import (
    TaskReport,
    cv_percent,
    render_reports,
    reports_to_dict,
    run_task,
)


def test_cv_percent_zero_for_constant():
    assert cv_percent([5.0, 5.0, 5.0]) == 0.0


def test_cv_percent_zero_for_single_value():
    assert cv_percent([42.0]) == 0.0


def test_cv_percent_zero_for_empty():
    assert cv_percent([]) == 0.0


def test_cv_percent_known_value():
    # values [1,2,3,4,5]: pstdev ~1.4142, mean 3 -> ~47.1%
    assert cv_percent([1, 2, 3, 4, 5]) == pytest.approx(47.14, abs=0.1)


def test_cv_percent_mean_zero_is_zero():
    assert cv_percent([-1.0, 1.0]) == 0.0


def test_run_task_collects_metrics():
    seq = iter([{"latency_ms": 10.0}, {"latency_ms": 20.0}, {"latency_ms": 30.0}])
    report = run_task("demo", lambda: next(seq), runs=3)
    summary = report.summary()
    assert summary["latency_ms"]["n"] == 3.0
    assert summary["latency_ms"]["mean"] == pytest.approx(20.0)
    assert summary["latency_ms"]["cv_percent"] > 0


def test_render_and_dict():
    rep = TaskReport(label="recall")
    for v in (1.0, 2.0, 3.0):
        rep.add("hits", v)
    text = render_reports([rep])
    assert "recall" in text and "hits" in text and "cv%" in text
    d = reports_to_dict([rep])
    assert d["recall"]["hits"]["n"] == 3.0
