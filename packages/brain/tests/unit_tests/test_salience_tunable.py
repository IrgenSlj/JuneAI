"""Runtime-tunable salience weights: config store + env override (S5.4)."""

from __future__ import annotations

import os

import pytest
from june_brain.memory.salience import SalienceWeights


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr("june_brain.config_store.MEMORY_DIR", str(tmp_path))
    for var in ("JUNE_SALIENCE_REL", "JUNE_SALIENCE_REC", "JUNE_SALIENCE_FREQ"):
        monkeypatch.delenv(var, raising=False)
    yield tmp_path


def test_load_defaults_when_unset(isolated_config):
    w = SalienceWeights.load()
    assert (w.rel, w.rec, w.freq) == pytest.approx((0.6, 0.25, 0.15))


def test_config_store_overrides_defaults(isolated_config):
    from june_brain.config_store import get_salience_weights, set_salience_weights

    set_salience_weights(rel=0.8, rec=0.1, freq=0.1)
    assert get_salience_weights() == pytest.approx({"rel": 0.8, "rec": 0.1, "freq": 0.1})

    w = SalienceWeights.load()
    assert (w.rel, w.rec, w.freq) == pytest.approx((0.8, 0.1, 0.1))


def test_env_overrides_config_store(isolated_config, monkeypatch):
    from june_brain.config_store import set_salience_weights

    set_salience_weights(rel=0.8, rec=0.1, freq=0.1)
    monkeypatch.setenv("JUNE_SALIENCE_REL", "0.95")

    w = SalienceWeights.load()
    # Env wins for rel; config still supplies rec/freq.
    assert w.rel == pytest.approx(0.95)
    assert w.rec == pytest.approx(0.1)
    assert w.freq == pytest.approx(0.1)


def test_partial_config_merges_with_defaults(isolated_config):
    from june_brain.config_store import set_salience_weights

    set_salience_weights(rel=0.7)  # only rel set
    w = SalienceWeights.load()
    assert w.rel == pytest.approx(0.7)
    assert w.rec == pytest.approx(0.25)  # default
    assert w.freq == pytest.approx(0.15)  # default


def test_set_weights_preserves_other_config(isolated_config):
    import json

    from june_brain.config_store import config_path, set_salience_weights

    config_path().write_text(json.dumps({"provider": "local"}), encoding="utf-8")
    set_salience_weights(rel=0.5)
    raw = json.loads(config_path().read_text(encoding="utf-8"))
    assert raw["provider"] == "local"  # untouched
    assert raw["salience_rel"] == 0.5


def test_env_only_still_works(isolated_config, monkeypatch):
    # Back-compat: from_env path unaffected.
    monkeypatch.setenv("JUNE_SALIENCE_FREQ", "0.3")
    assert SalienceWeights.load().freq == pytest.approx(0.3)
    assert os.environ["JUNE_SALIENCE_FREQ"] == "0.3"
