"""Population sampler tests (D6.4 / D5.7).

The ~50k synthetic population must be a faithful *distribution anchor* of the real 1,600: the fitted
static marginals + the three joints (plan x spend, plan x age, plan x region) are preserved, the
1,600 anchor rows keep their REAL static attributes verbatim (only health/events/label are
re-simulated downstream), and the whole draw is a pure function of the seed. No label information
enters here (invariant i).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn import config
from churn.simulator import params as P
from churn.simulator import population as POP

STATIC_COLS = ["region", "device_type", "subscription_plan", *P.STATIC_NUMERICS]


@pytest.fixture(scope="module")
def params() -> dict:
    return P.load_params()


@pytest.fixture(scope="module")
def anchor() -> pd.DataFrame:
    return pd.read_csv(config.DATA_PATH)


@pytest.fixture(scope="module")
def pop(params) -> pd.DataFrame:
    return POP.sample_population(params, seed=101, n_synthetic=8000)


def test_size_and_anchor_partition(pop, params):
    assert len(pop) == params["population"]["anchor_n"] + 8000
    assert int(pop["is_anchor"].sum()) == params["population"]["anchor_n"]
    assert pop["customer_id"].is_unique


def test_anchor_rows_keep_real_static_verbatim(pop, anchor):
    a = pop[pop["is_anchor"]].reset_index(drop=True)
    ref = anchor.reset_index(drop=True)
    assert len(a) == len(ref)
    for col in STATIC_COLS:
        if col in P.STATIC_NUMERICS:
            assert np.allclose(a[col].to_numpy(float), ref[col].to_numpy(float), equal_nan=True)
        else:
            assert a[col].tolist() == ref[col].tolist()


def test_anchor_carries_real_label_but_synthetic_does_not(pop):
    a = pop[pop["is_anchor"]]
    s = pop[~pop["is_anchor"]]
    assert a["real_churned"].notna().all()
    assert set(a["real_churned"].unique()) <= {0, 1}
    assert s["real_churned"].isna().all()  # synthetic have no real outcome (invariant i)


def test_plan_marginal_is_preserved(pop, anchor):
    syn = pop[~pop["is_anchor"]]
    a = anchor["subscription_plan"].value_counts(normalize=True)
    s = syn["subscription_plan"].value_counts(normalize=True)
    for plan in a.index:
        assert s.get(plan, 0.0) == pytest.approx(a[plan], abs=0.03)


def test_device_marginal_is_preserved(pop, anchor):
    syn = pop[~pop["is_anchor"]]
    a = anchor["device_type"].value_counts(normalize=True)
    s = syn["device_type"].value_counts(normalize=True)
    for dev in a.index:
        assert s.get(dev, 0.0) == pytest.approx(a[dev], abs=0.03)


def test_plan_x_spend_joint_is_preserved(pop, anchor):
    # The plan x spend joint is checked with the ROBUST within-plan median: monthly_spend is heavy-
    # tailed (median 56 vs mean 243), so the mean is a high-variance statistic and the median is the
    # faithful "distribution preserved" check the multiplicative jitter is designed to protect.
    syn = pop[~pop["is_anchor"]]
    for plan, grp in anchor.groupby("subscription_plan"):
        ref = float(grp["monthly_spend"].median())
        got = float(syn[syn["subscription_plan"] == plan]["monthly_spend"].median())
        assert got == pytest.approx(ref, rel=0.20)


def test_plan_x_aov_joint_is_preserved(pop, anchor):
    # avg_order_value is drawn plan-conditionally (D6.4 Codex), so its within-plan median tracks.
    syn = pop[~pop["is_anchor"]]
    for plan, grp in anchor.groupby("subscription_plan"):
        ref = float(grp["avg_order_value"].median())
        got = float(syn[syn["subscription_plan"] == plan]["avg_order_value"].median())
        assert got == pytest.approx(ref, rel=0.20)


def test_plan_x_region_joint_is_preserved(pop, anchor):
    # Region composition within a plan tracks the anchor's plan x region joint.
    syn = pop[~pop["is_anchor"]]
    for plan in anchor["subscription_plan"].unique():
        ref = anchor[anchor["subscription_plan"] == plan]["region"].value_counts(normalize=True)
        got = syn[syn["subscription_plan"] == plan]["region"].value_counts(normalize=True)
        for region in ref.index:
            assert got.get(region, 0.0) == pytest.approx(ref[region], abs=0.08)


def test_determinism_and_ranges(params):
    a = POP.sample_population(params, seed=7, n_synthetic=2000)
    b = POP.sample_population(params, seed=7, n_synthetic=2000)
    pd.testing.assert_frame_equal(a, b)
    syn = a[~a["is_anchor"]]
    assert syn[P.STATIC_NUMERICS].notna().all().all()
    assert (syn["account_age_days"] >= 0).all() and (syn["account_age_days"] <= 3650).all()
    assert (syn["monthly_spend"] >= 0).all() and (syn["avg_order_value"] >= 0).all()


def test_synthetic_ids_are_opaque_and_disjoint_from_anchor(pop, anchor):
    syn_ids = set(pop[~pop["is_anchor"]]["customer_id"])
    anchor_ids = set(anchor["customer_id"].astype(str))
    assert syn_ids.isdisjoint(anchor_ids)
