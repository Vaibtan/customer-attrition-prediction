"""Online store tests -- versioned envelope round-trip, strict contract, store-level parity (§3).

The online store must round-trip a *stamped* feature envelope (ADR 0007) and -- when fed by the
streaming aggregation -- serve the SAME vector the offline DuckDB PIT query produces for the same
``(customer, t0)``. The reader is strict: schema/as_of mismatch, an incomplete vector, or a
pre-envelope blob raises ``FeatureContractError`` -- a parity score is real or refused.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pandas as pd
import pytest

from churn.featurestore import offline as OFF
from churn.featurestore import online as ON
from churn.featurestore.cohort import make_cohort
from churn.simulator import generate as G
from churn.simulator import params as P
from churn.streaming import aggregate as AGG

T0 = pd.Timestamp("2025-01-01T00:00:00")


def _full_vector(fill: float = 1.0) -> dict[str, float]:
    return dict.fromkeys(OFF.FEATURE_COLUMNS, fill)


def test_store_round_trips_a_stamped_envelope():
    store = ON.OnlineStore()
    store.put("A", _full_vector(3.0), as_of=T0)
    envelope = store.get("A")
    assert envelope is not None
    assert envelope.schema == ON.FEATURE_SCHEMA
    assert envelope.as_of == T0.isoformat()
    assert envelope.features == _full_vector(3.0)
    assert store.get("MISSING") is None
    ON.validate_envelope(envelope, T0)  # a fresh write validates clean


def test_schema_is_derived_from_the_feature_columns():
    """Change the spec -> the hash changes by construction; no manual bump to forget."""
    import hashlib  # noqa: PLC0415

    expected = hashlib.sha256("|".join(OFF.FEATURE_COLUMNS).encode("ascii")).hexdigest()[:12]
    assert ON.FEATURE_SCHEMA == expected


def test_validate_rejects_schema_mismatch_as_of_mismatch_and_incomplete():
    store = ON.OnlineStore()
    store.put("A", _full_vector(), as_of=T0)
    envelope = store.get("A")

    stale = dataclasses.replace(envelope, schema="deadbeef0000")
    with pytest.raises(ON.FeatureContractError, match="schema mismatch"):
        ON.validate_envelope(stale, T0)

    with pytest.raises(ON.FeatureContractError, match="as_of mismatch"):
        ON.validate_envelope(envelope, T0 + pd.Timedelta(days=7))

    partial = dict(list(_full_vector().items())[:3])
    incomplete = dataclasses.replace(envelope, features=partial)
    with pytest.raises(ON.FeatureContractError, match="incomplete"):
        ON.validate_envelope(incomplete, T0)


def test_nan_features_fail_loudly_at_the_producer():
    """allow_nan=False: a NaN never lands in the store (REV-04's blast radius, bounded)."""
    store = ON.OnlineStore()
    vec = _full_vector()
    vec["mean_session_depth_28d"] = math.nan
    with pytest.raises(ValueError):
        store.put("A", vec, as_of=T0)
    assert store.get("A") is None  # nothing was written


def test_pre_envelope_blob_is_a_contract_error_not_a_silent_score():
    store = ON.OnlineStore()
    store.backend.set("feat:LEGACY", json.dumps({"login_count_90d": 3.0}))
    with pytest.raises(ON.FeatureContractError, match="unversioned"):
        store.get("LEGACY")


def test_online_store_serves_the_offline_pit_vector():
    ds = G.build_population_dataset(P.load_params(), seed=4242, n_synthetic=0)
    ids = ds.customers["customer_id"].head(30).tolist()
    events = ds.events[ds.events["customer_id"].isin(ids)]
    cohort = make_cohort(ids, T0)

    offline = OFF.compute_pit_features(events, cohort).set_index("customer_id")
    online_vecs = AGG.aggregate_stream(events, {c: T0 for c in ids})
    store = ON.OnlineStore()
    store.put_many(online_vecs, as_of=T0)

    for c in ids:
        envelope = store.get(c)
        ON.validate_envelope(envelope, T0)
        for col in OFF.FEATURE_COLUMNS:
            assert envelope.features[col] == pytest.approx(float(offline.loc[c, col]), abs=1e-9)


def test_put_many_and_missing_key():
    store = ON.OnlineStore()
    store.put_many({"A": _full_vector(1.0), "B": _full_vector(2.0)}, as_of=T0)
    assert store.get("A").features["login_count_90d"] == 1.0
    assert store.get("B").features["login_count_90d"] == 2.0
    assert store.get("C") is None


class _SpyBackend:
    """Records call counts so put_many's ONE-round-trip contract is asserted, not assumed."""

    def __init__(self) -> None:
        self._d: dict[str, str] = {}
        self.set_calls = 0
        self.set_many_calls = 0

    def set(self, key: str, value: str) -> None:
        self.set_calls += 1
        self._d[key] = value

    def set_many(self, items: dict[str, str]) -> None:
        self.set_many_calls += 1
        self._d.update(items)

    def get(self, key: str) -> str | None:
        return self._d.get(key)


def test_put_many_makes_one_backend_round_trip():
    """ADR 0008: put_many batches transport -- one set_many/MSET, never N SETs."""
    spy = _SpyBackend()
    store = ON.OnlineStore(spy)
    store.put_many(
        {"A": _full_vector(1.0), "B": _full_vector(2.0), "C": _full_vector(3.0)}, as_of=T0
    )
    assert spy.set_many_calls == 1  # one round trip for the whole batch
    assert spy.set_calls == 0  # not one SET per customer
    assert store.get("B").features["login_count_90d"] == 2.0  # round-trips correctly


def test_put_many_is_all_or_nothing_on_a_nan_vector():
    """Envelopes are built BEFORE the backend call: one NaN aborts the whole MSET (no partial
    write). allow_nan=False must fire at the producer for a batch just as for a single put."""
    spy = _SpyBackend()
    store = ON.OnlineStore(spy)
    bad = _full_vector(1.0)
    bad["mean_session_depth_28d"] = math.nan
    with pytest.raises(ValueError):
        store.put_many({"A": _full_vector(1.0), "B": bad}, as_of=T0)
    assert spy.set_many_calls == 0  # nothing reached the backend
    assert store.get("A") is None  # not even the good vector was written


def test_put_many_empty_is_a_noop():
    """An empty batch must not reach the backend (redis MSET rejects an empty mapping)."""
    spy = _SpyBackend()
    ON.OnlineStore(spy).put_many({}, as_of=T0)
    assert spy.set_many_calls == 0 and spy.set_calls == 0
