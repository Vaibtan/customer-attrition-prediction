"""Smoke test for the scalar-MC tuning harness (D5.9).

Drives the whole generation pipeline at tiny N against a tmp path (so the committed
simulator.params.json is never touched), covering the search + back-out + write code. The
*frozen* regime is checked precisely in test_simulator_params.py; here we only assert the
generator runs and produces a structurally valid, non-degenerate world.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from churn.simulator import tuning as T


def test_helpers_are_wired(tmp_path):
    static = T.load_anchor_static()
    assert len(static) == 1600
    stats = T.numeric_stats(static)
    assert set(stats) == set(T.STATIC_NUMERICS)
    q_raw = T.raw_quality(static, stats)
    assert q_raw.shape == (1600,)


def test_bisect_finds_a_monotone_root():
    root = T._bisect(lambda x: x, 0.0, 10.0, target=3.0, iters=40)
    assert root == pytest.approx(3.0, abs=1e-6)


def test_tuning_main_regenerates_a_valid_world(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "SEARCH_N", 8000)
    monkeypatch.setattr(T, "REPORT_N", 8000)
    monkeypatch.setattr(T, "SIM_PARAMS_PATH", tmp_path / "simulator.params.json")
    monkeypatch.setattr(T, "TUNING_REPORT_PATH", tmp_path / "reports" / "simulator_tuning.json")

    T.main()

    params = json.loads((tmp_path / "simulator.params.json").read_text())
    assert params["schema_version"] == 1
    assert {"latent", "quality_index", "hazard", "events", "controls"} <= params.keys()
    # a-priori event params are copied through unchanged (D7 re-lock value)
    assert params["events"]["login_rate"]["b"] == 0.7
    # tuned hazard coefficients carry the frozen sign convention (churn falls with h and q)
    assert params["hazard"]["alpha_h"] < 0 and params["hazard"]["alpha_stat"] < 0

    report = json.loads((tmp_path / "reports" / "simulator_tuning.json").read_text())
    regime = report["achieved_regime"]
    # Even at tiny N the world is non-degenerate: real headroom, roughly balanced base rate.
    assert 0.40 <= regime["base_rate"] <= 0.56
    assert regime["oracle_auc"] - regime["static_auc"] >= 0.08


def test_evaluate_regime_round_trips(monkeypatch):
    from churn.simulator import params as P

    frozen = P.load_params()
    static = T.load_anchor_static()
    num_stats = T.numeric_stats(static)
    q_raw = T.raw_quality(static, num_stats)
    q_anchor = (q_raw - q_raw.mean()) / q_raw.std(ddof=0)
    regime = T.evaluate_regime(frozen, q_anchor, np.random.default_rng(3), n=20_000)
    assert 0.60 <= regime["static_auc"] <= 0.68
    assert 0.76 <= regime["oracle_auc"] <= 0.84
