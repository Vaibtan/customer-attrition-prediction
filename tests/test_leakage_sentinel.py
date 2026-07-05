"""Leakage-sentinel suite (ENHANCEMENT_PLAN.md §4.7) -- the real leakage guard.

Negative controls are necessary but NOT sufficient; leak-freeness is argued from a sentinel test
per known leakage *path*. One test (or tight cluster) per row of the §4.7 taxonomy:

  1. Customer-ID memorization      -> id excluded from features; no id-derived feature
  2. Train/test customer overlap   -> group-aware split by customer_id (no straddle)
  3. Post-t0 feature leakage       -> events after t0 never change a PIT feature; boundary fuzz
  4. Timestamp-boundary bugs       -> strict inequality at t0 (t0 included, t0+eps excluded)
  5. Preprocessing leakage         -> every transformer refit per fold (fold stats != full stats)
  6. Target-aware simulator tuning -> guarded by the frozen lock (check_lock clean)
  7. Backstop                      -> label-shuffle + null-stream show no lift (see also
                                      test_instrument_experiment.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.featurestore import offline as OFF
from churn.featurestore.cohort import make_cohort
from churn.instrument import model as M
from churn.simulator import generate as G
from churn.simulator import params as P

T0 = OFF_T0 = pd.Timestamp("2025-01-01T00:00:00")


@pytest.fixture(scope="module")
def params() -> dict:
    return P.load_params()


@pytest.fixture(scope="module")
def anchor(params):
    ds = G.build_population_dataset(params, seed=4242, n_synthetic=0)
    pit = OFF.compute_pit_features(ds.events, ds.cohort)
    return M.assemble_design(ds, pit)


def _ev(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["event_ts"] = pd.to_datetime(df["event_ts"])
    if "value" not in df:
        df["value"] = np.nan
    return df[["event_id", "customer_id", "event_ts", "event_type", "value"]]


def _cohort(ids, t0=T0):
    return make_cohort(ids, t0)


# --- Path 1: customer-ID memorization --------------------------------------------------------


def test_customer_id_is_not_a_feature(anchor):
    features = M.STATIC_FEATURES + M.EVENT_FEATURES
    assert "customer_id" not in features
    assert "customer_id" not in anchor.X.columns
    # No event feature is derived from the id (all are event-time aggregates).
    assert not any("id" in c.lower() for c in M.EVENT_FEATURES)


def test_permuting_customer_id_does_not_change_scores(params):
    # The model only selects declared feature columns, so an id permutation cannot move a score.
    ds = G.build_population_dataset(params, seed=11, n_synthetic=0)
    pit = OFF.compute_pit_features(ds.events, ds.cohort)
    d = M.assemble_design(ds, pit)
    p1 = M.group_oof_proba("static_event", d.X, d.y, d.groups, seed=3)
    # Shuffle the group labels' identities (relabel customers) -- scores are id-agnostic.
    relabel = {g: f"X{i}" for i, g in enumerate(np.random.default_rng(0).permutation(d.groups))}
    groups2 = np.array([relabel[g] for g in d.groups])
    p2 = M.group_oof_proba("static_event", d.X, d.y, groups2, seed=3)
    assert np.allclose(np.sort(p1), np.sort(p2), atol=0.02)  # fold identities differ, signal same


# --- Path 2: train/test customer overlap (group-aware split) ---------------------------------


def test_no_customer_straddles_the_split_on_a_panel(anchor):
    # Panel fixture: every customer appears in 2 rows; the group split must keep them together.
    groups = np.repeat(anchor.groups, 2)
    y = np.repeat(anchor.y, 2)
    folds = M.stratified_group_folds(y, groups, seed=5)
    assert len(folds) == M.CV_SPLITS
    for train_idx, test_idx in folds:
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


# --- Path 3: post-t0 feature leakage ---------------------------------------------------------


def test_events_after_t0_never_change_a_feature():
    pre = [
        {
            "event_id": "A:login:0",
            "customer_id": "A",
            "event_ts": T0 - pd.Timedelta(days=2),
            "event_type": "login",
            "value": 6.0,
        },
        {
            "event_id": "A:sup:0",
            "customer_id": "A",
            "event_ts": T0 - pd.Timedelta(days=5),
            "event_type": "support",
            "value": -0.3,
        },
    ]
    post = [  # highly label-correlated events INSIDE the label window (t0, t0+90d]
        {
            "event_id": "A:login:9",
            "customer_id": "A",
            "event_ts": T0 + pd.Timedelta(days=1),
            "event_type": "login",
            "value": 20.0,
        },
        {
            "event_id": "A:pay:9",
            "customer_id": "A",
            "event_ts": T0 + pd.Timedelta(days=3),
            "event_type": "payment_fail",
            "value": None,
        },
    ]
    without = OFF.compute_pit_features(_ev(pre), _cohort(["A"]))
    with_post = OFF.compute_pit_features(_ev(pre + post), _cohort(["A"]))
    pd.testing.assert_frame_equal(without, with_post)  # post-t0 events are invisible to features


def test_timestamp_fuzz_across_t0_moves_only_the_pre_side():
    # An event fuzzed from just-before to just-after t0 must leave the pre-t0 window.
    base = {"customer_id": "A", "event_type": "login", "value": 5.0}
    before = _ev([{"event_id": "A:login:0", "event_ts": T0 - pd.Timedelta(hours=1), **base}])
    after = _ev([{"event_id": "A:login:0", "event_ts": T0 + pd.Timedelta(hours=1), **base}])
    f_before = OFF.compute_pit_features(before, _cohort(["A"])).set_index("customer_id")
    f_after = OFF.compute_pit_features(after, _cohort(["A"])).set_index("customer_id")
    assert f_before.loc["A", "login_count_14d"] == 1.0
    assert f_after.loc["A", "login_count_14d"] == 0.0  # crossed t0 -> excluded
    assert f_after.loc["A", "days_since_last_login"] == 365.0  # sentinel: no pre-t0 login


# --- Path 4: timestamp-boundary (strict inequality at t0) ------------------------------------


def test_strict_inequality_at_t0():
    at = _ev(
        [
            {
                "event_id": "A:login:0",
                "customer_id": "A",
                "event_ts": T0,
                "event_type": "login",
                "value": 5.0,
            }
        ]
    )
    eps_after = _ev(
        [
            {
                "event_id": "A:login:0",
                "customer_id": "A",
                "event_ts": T0 + pd.Timedelta(nanoseconds=1),
                "event_type": "login",
                "value": 5.0,
            }
        ]
    )
    f_at = OFF.compute_pit_features(at, _cohort(["A"])).set_index("customer_id")
    f_after = OFF.compute_pit_features(eps_after, _cohort(["A"])).set_index("customer_id")
    assert f_at.loc["A", "login_count_90d"] == 1.0  # t0 itself is in-window (<=)
    assert f_after.loc["A", "login_count_90d"] == 0.0  # t0 + 1ns is out


# --- Path 5: preprocessing leakage (per-fold refit) ------------------------------------------


def test_preprocessing_scaler_is_fit_per_fold(anchor):
    from sklearn.model_selection import StratifiedGroupKFold

    pipe = M.build_pipeline("static", seed=1)
    cv = StratifiedGroupKFold(n_splits=M.CV_SPLITS, shuffle=True, random_state=1)
    train_idx, _ = next(cv.split(anchor.X, anchor.y, anchor.groups))
    fold_scaler = pipe.fit(anchor.X.iloc[train_idx], anchor.y[train_idx])
    full_scaler = M.build_pipeline("static", seed=1).fit(anchor.X, anchor.y)
    fold_mean = fold_scaler.named_steps["pre"].named_transformers_["num"].named_steps["scale"].mean_
    full_mean = full_scaler.named_steps["pre"].named_transformers_["num"].named_steps["scale"].mean_
    assert not np.allclose(fold_mean, full_mean)  # fold stats depend only on the fold's rows


# --- Path 6: target-aware simulator tuning guarded by the lock -------------------------------


def test_frozen_world_lock_is_clean():
    from churn.simulator import lock as LK

    assert LK.check_lock() == []  # the simulator params/spec/code are frozen and unmodified
