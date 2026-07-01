"""Frozen-params tests: schema, the kernel<->frozen-params binding, and world non-degeneracy.

These complement the pure-formula golden vectors in test_simulator_kernels.py:
  * binding golden vectors evaluate each event family at its FROZEN (a, b) and check the
    hand-computed baseline semantics (e.g. ~4 logins/wk, ~5% payment failure at h=0);
  * the non-degeneracy sanity test re-derives the regime from the frozen params via an
    INDEPENDENT MC path (params.quality_index + kernels.hazard_prob, fresh seed) and asserts
    the world has real recoverable headroom (oracle_auc >> static_auc). This is the Phase-0
    check that "tune-then-freeze" (D5.9) actually froze a non-degenerate world.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from churn import config
from churn.simulator import kernels as K
from churn.simulator import params as P


@pytest.fixture(scope="module")
def params() -> dict:
    return P.load_params()


@pytest.fixture(scope="module")
def anchor_static() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PATH)
    return df[
        [
            "region",
            "device_type",
            "subscription_plan",
            "account_age_days",
            "monthly_spend",
            "avg_order_value",
        ]
    ].copy()


# --- schema -----------------------------------------------------------------------------------


def test_params_load_and_have_no_confirmatory_seed_in_the_clear(params):
    assert params["schema_version"] == 1
    # D3: the confirmatory seed is never committed in the clear at Stage 1.
    blob = repr(params).lower()
    assert "confirmatory_seed" not in blob and "confirmatory-seed" not in blob


def test_load_params_rejects_missing_keys(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text('{"schema_version": 1}')
    with pytest.raises(ValueError, match="missing keys"):
        P.load_params(bad)


def test_load_params_rejects_unknown_schema_version(tmp_path, params):
    import json

    bumped = {**params, "schema_version": 99}
    bad = tmp_path / "p.json"
    bad.write_text(json.dumps(bumped))
    with pytest.raises(ValueError, match="schema_version"):
        P.load_params(bad)


def test_committed_params_match_tuning_source_constants(params):
    """The a-priori constants in tuning.py must equal the committed (hashed) params.json copy,
    so the generator and the frozen artifact cannot silently diverge (adversarial-review gap)."""
    from churn.simulator import tuning as T

    assert params["events"] == T.EVENT_PARAMS
    assert params["quality_index"]["weights"] == T.QI_WEIGHTS
    assert params["quality_index"]["plan_score_map"] == T.PLAN_SCORE_MAP
    assert params["floor_design"] == T.FLOOR_DESIGN


# --- binding golden vectors: each event family at its FROZEN (a, b) (D5.5 / D5.10) ------------


def test_event_family_baselines_match_the_frozen_design(params):
    ev = params["events"]
    # At h = 0 each link returns its designed baseline (the a-priori semantics).
    assert K.exp_link(0.0, **_ab(ev["login_rate"])) == pytest.approx(4.0)  # ~4 logins/wk
    assert K.exp_link(0.0, **_ab(ev["order_rate"])) == pytest.approx(0.5)  # ~0.5 orders/wk
    assert K.exp_link(0.0, **_ab(ev["support_rate"])) == pytest.approx(0.15)  # ~0.15 tickets/wk
    assert K.logistic_link(0.0, **_ab(ev["payment_fail"])) == pytest.approx(0.05)  # ~5% failure
    assert K.logistic_link(0.0, **_ab(ev["downgrade"])) == pytest.approx(0.02)  # ~2% downgrade
    assert K.linear_link(0.0, **_ab(ev["session_depth"])) == pytest.approx(8.0)  # ~8 pages
    assert K.linear_link(0.0, **_ab(ev["sentiment"])) == pytest.approx(0.0)  # neutral


def test_frozen_event_directions(params):
    ev = params["events"]
    # Healthier -> more logins; unhealthier -> more payment failures.
    assert K.exp_link(1.0, **_ab(ev["login_rate"])) > K.exp_link(-1.0, **_ab(ev["login_rate"]))
    assert K.exp_link(1.0, **_ab(ev["login_rate"])) == pytest.approx(4.0 * math.exp(0.4))
    assert K.logistic_link(-1.0, **_ab(ev["payment_fail"])) > K.logistic_link(
        1.0, **_ab(ev["payment_fail"])
    )


def _ab(d: dict) -> dict:
    return {"a": d["a"], "b": d["b"]}


# --- quality index binding (D5.4) ------------------------------------------------------------


def test_quality_index_is_standardized_on_the_anchor(params, anchor_static):
    q = P.quality_index(anchor_static, params)
    assert q.mean() == pytest.approx(0.0, abs=1e-6)
    assert q.std(ddof=0) == pytest.approx(1.0, abs=1e-6)


def test_quality_index_is_monotone_in_plan_tier(params):
    # Free -> Enterprise should raise q (healthier), holding numerics at their medians.
    ns = params["quality_index"]["numeric_standardize"]
    rows = pd.DataFrame(
        {
            "region": ["North"] * 4,
            "device_type": ["Mobile"] * 4,
            "subscription_plan": ["Free", "Basic", "Premium", "Enterprise"],
            "account_age_days": [ns["account_age_days"]["median"]] * 4,
            "monthly_spend": [ns["monthly_spend"]["median"]] * 4,
            "avg_order_value": [ns["avg_order_value"]["median"]] * 4,
        }
    )
    q = P.quality_index(rows, params)
    assert list(q) == sorted(q)  # strictly increasing in plan tier


# --- non-degeneracy sanity check (D5.9) ------------------------------------------------------


def test_frozen_world_has_real_recoverable_headroom(params, anchor_static):
    """Independent MC from the frozen params: oracle_auc must sit well above static_auc."""
    rng = np.random.default_rng(7)  # fresh seed, distinct from the tuning dev seed
    n = 40_000
    q_anchor = P.quality_index(anchor_static, params)
    idx = rng.integers(0, len(q_anchor), size=n)
    q = q_anchor[idx]

    haz = params["hazard"]
    s_mu = params["quality_index"]["s_mu"]
    h = s_mu * q + params["latent"]["sigma0"] * rng.standard_normal(n)
    p = K.hazard_prob(h, q, haz["alpha0"], haz["alpha_h"], haz["alpha_stat"])
    y = (rng.random(n) < p).astype(int)

    base_rate = y.mean()
    static_auc = roc_auc_score(y, -q)  # higher q -> lower churn
    oracle_auc = roc_auc_score(y, p)

    assert 0.45 <= base_rate <= 0.51, base_rate
    assert 0.61 <= static_auc <= 0.67, static_auc
    assert 0.77 <= oracle_auc <= 0.83, oracle_auc
    assert oracle_auc - static_auc >= 0.12, oracle_auc - static_auc  # real event headroom
