"""Online/offline parity (ENHANCEMENT_PLAN.md §3, §4.4 -- how Quix Streams earns its place).

The streaming event-time aggregation (an INDEPENDENT Python implementation of the online feature
state) must produce the byte-identical feature vector to the offline DuckDB PIT query for the same
``(customer, t0)`` -- **under adversarial streams**: late, duplicate, reordered, and boundary
events. This is the exact train/serve-skew failure mode the platform exists to prevent; the parity
is the deliverable, not the broker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.featurestore import offline as OFF
from churn.featurestore.cohort import make_cohort
from churn.simulator import generate as G
from churn.simulator import params as P
from churn.streaming import aggregate as AGG

T0 = OFF_T0 = pd.Timestamp("2025-01-01T00:00:00")


def _ev(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["event_ts"] = pd.to_datetime(df["event_ts"])
    if "value" not in df:
        df["value"] = np.nan
    return df[["event_id", "customer_id", "event_ts", "event_type", "value"]]


def _assert_parity(events: pd.DataFrame, ids: list[str], t0=T0):
    cohort = make_cohort(ids, t0)
    offline = OFF.compute_pit_features(events, cohort).set_index("customer_id")
    # Both online reducers must equal offline: the recompute reference AND the O(1)-per-event
    # incremental path the deployed consumer actually runs (SQL == recompute == incremental).
    recompute = AGG.aggregate_stream(events, {c: t0 for c in ids})
    incremental = AGG.aggregate_stream_incremental(events, {c: t0 for c in ids})
    for c in ids:
        off_row = offline.loc[c]
        for col in OFF.FEATURE_COLUMNS:
            target = pytest.approx(float(off_row[col]), abs=1e-9)
            assert recompute[c][col] == target, ("recompute", c, col)
            assert incremental[c][col] == target, ("incremental", c, col)


def test_parity_on_a_real_customer_stream():
    ds = G.build_population_dataset(P.load_params(), seed=4242, n_synthetic=0)
    ids = ds.customers["customer_id"].head(40).tolist()
    events = ds.events[ds.events["customer_id"].isin(ids)]
    _assert_parity(events, ids)


def test_parity_under_reordering():
    ds = G.build_population_dataset(P.load_params(), seed=7, n_synthetic=0)
    ids = ds.customers["customer_id"].head(25).tolist()
    events = ds.events[ds.events["customer_id"].isin(ids)]
    # arrival order != event time
    shuffled = events.sample(frac=1.0, random_state=1).reset_index(drop=True)
    _assert_parity(shuffled, ids)


def test_parity_under_duplicates():
    events = _ev(
        [
            {
                "event_id": "A:login:0",
                "customer_id": "A",
                "event_ts": T0 - pd.Timedelta(days=3),
                "event_type": "login",
                "value": 6.0,
            },
            {
                "event_id": "A:login:1",
                "customer_id": "A",
                "event_ts": T0 - pd.Timedelta(days=20),
                "event_type": "login",
                "value": 4.0,
            },
            {
                "event_id": "A:sup:0",
                "customer_id": "A",
                "event_ts": T0 - pd.Timedelta(days=10),
                "event_type": "support",
                "value": -0.4,
            },
        ]
    )
    dup = pd.concat([events, events.iloc[[0, 2]], events.iloc[[0]]], ignore_index=True)
    _assert_parity(dup, ["A"])


def test_parity_under_late_and_post_t0_events():
    events = _ev(
        [
            {
                "event_id": "A:login:0",
                "customer_id": "A",
                "event_ts": T0 - pd.Timedelta(days=2),
                "event_type": "login",
                "value": 7.0,
            },
            # a "late" event (old event arriving last) + a post-t0 event (must be ignored).
            {
                "event_id": "A:login:1",
                "customer_id": "A",
                "event_ts": T0 - pd.Timedelta(days=200),
                "event_type": "login",
                "value": 3.0,
            },
            {
                "event_id": "A:login:9",
                "customer_id": "A",
                "event_ts": T0 + pd.Timedelta(days=1),
                "event_type": "login",
                "value": 99.0,
            },
        ]
    )
    _assert_parity(events, ["A"])


def test_parity_at_the_t0_boundary():
    events = _ev(
        [
            {
                "event_id": "A:login:0",
                "customer_id": "A",
                "event_ts": T0,
                "event_type": "login",
                "value": 5.0,
            },  # exactly at t0 -> included
            {
                "event_id": "A:login:1",
                "customer_id": "A",
                "event_ts": T0 + pd.Timedelta(nanoseconds=1),
                "event_type": "login",
                "value": 5.0,
            },  # t0+1ns -> excluded
        ]
    )
    _assert_parity(events, ["A"])


def test_parity_for_empty_customer():
    events = _ev(
        [
            {
                "event_id": "A:login:0",
                "customer_id": "A",
                "event_ts": T0 - pd.Timedelta(days=1),
                "event_type": "login",
                "value": 5.0,
            },
        ]
    )
    _assert_parity(events, ["A", "GHOST"])  # GHOST has no events -> both sides emit the sentinels


def test_incremental_aggregator_state_round_trips():
    # The consumer persists/reloads the aggregator as a JSON-friendly blob every event; a
    # round-trip through to_state/from_state must not change the emitted vector.
    agg = AGG.CustomerAggregator(t0_ns=int(T0.value))
    for days, typ, val in [(3, "login", 6.0), (20, "login", 4.0), (10, "support", -0.4)]:
        agg.add(int((T0 - pd.Timedelta(days=days)).value), typ, val)
    restored = AGG.CustomerAggregator.from_state(agg.to_state())
    assert restored.features() == agg.features()
    # ... and it equals the recompute reducer on the same (deduped) events.
    events = {
        "A:login:0": (T0 - pd.Timedelta(days=3), "login", 6.0),
        "A:login:1": (T0 - pd.Timedelta(days=20), "login", 4.0),
        "A:sup:0": (T0 - pd.Timedelta(days=10), "support", -0.4),
    }
    recompute = AGG.feature_vector(events, T0)
    for col in OFF.FEATURE_COLUMNS:
        assert agg.features()[col] == pytest.approx(recompute[col], abs=1e-9)
