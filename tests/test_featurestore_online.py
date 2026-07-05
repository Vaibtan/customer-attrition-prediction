"""Online store tests -- round-trip + the store-level online/offline parity (§3).

The online store must round-trip a feature vector and -- when fed by the streaming aggregation --
serve the SAME vector the offline DuckDB PIT query produces for the same ``(customer, t0)``. This is
the train/serve-consistency deliverable expressed at the store boundary.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn.featurestore import offline as OFF
from churn.featurestore import online as ON
from churn.featurestore.cohort import make_cohort
from churn.simulator import generate as G
from churn.simulator import params as P
from churn.streaming import aggregate as AGG

T0 = pd.Timestamp("2025-01-01T00:00:00")


def test_store_round_trips():
    store = ON.OnlineStore()
    store.put("A", {"login_count_90d": 3.0, "days_since_last_login": 5.0})
    assert store.get("A") == {"login_count_90d": 3.0, "days_since_last_login": 5.0}
    assert store.get("MISSING") is None


def test_online_store_serves_the_offline_pit_vector():
    ds = G.build_population_dataset(P.load_params(), seed=4242, n_synthetic=0)
    ids = ds.customers["customer_id"].head(30).tolist()
    events = ds.events[ds.events["customer_id"].isin(ids)]
    cohort = make_cohort(ids, T0)

    offline = OFF.compute_pit_features(events, cohort).set_index("customer_id")
    online_vecs = AGG.aggregate_stream(events, {c: T0 for c in ids})
    store = ON.OnlineStore()
    store.put_many(online_vecs)

    for c in ids:
        served = store.get(c)
        for col in OFF.FEATURE_COLUMNS:
            assert served[col] == pytest.approx(float(offline.loc[c, col]), abs=1e-9)


def test_put_many_and_missing_key():
    store = ON.OnlineStore()
    store.put_many({"A": {"x": 1.0}, "B": {"x": 2.0}})
    assert store.get("A")["x"] == 1.0 and store.get("B")["x"] == 2.0
    assert store.get("C") is None
