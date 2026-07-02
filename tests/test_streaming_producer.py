"""Producer wire-format unit tests (pure -- no broker, runs on the host).

The wire format is the train/serve contract's first link: event time must survive JSON transport
losslessly (int64 nanoseconds) and a missing magnitude (NaN) must round-trip as JSON null so the
online aggregation reconstructs the exact same event set the offline PIT query sees.
"""

from __future__ import annotations

import math

import pandas as pd

from churn.streaming import producer as PROD

T0 = pd.Timestamp("2025-01-01T00:00:00")


def test_serialize_round_trips_a_login_event_losslessly():
    rec = {
        "event_id": "A:login:0",
        "customer_id": "A",
        "event_ts": T0 - pd.Timedelta(days=3, minutes=17, seconds=42),
        "event_type": "login",
        "value": 6.0,
    }
    wire = PROD.deserialize_event(PROD.serialize_event(rec))
    assert wire["event_id"] == "A:login:0"
    assert wire["customer_id"] == "A"
    assert wire["event_type"] == "login"
    assert wire["value"] == 6.0
    # Event time survives as int64 ns and reconstructs the exact original instant.
    assert isinstance(wire["event_ts_ns"], int)
    assert pd.Timestamp(wire["event_ts_ns"]) == rec["event_ts"]


def test_nan_value_serialises_to_null():
    rec = {
        "event_id": "A:order:0",
        "customer_id": "A",
        "event_ts": T0,
        "event_type": "order",
        "value": float("nan"),
    }
    wire = PROD.deserialize_event(PROD.serialize_event(rec))
    assert wire["value"] is None


def test_serialize_is_deterministic_bytes():
    rec = {
        "event_id": "A:sup:0",
        "customer_id": "A",
        "event_ts": T0,
        "event_type": "support",
        "value": -0.4,
    }
    assert PROD.serialize_event(rec) == PROD.serialize_event(dict(rec))


def test_iter_records_normalises_a_dataframe():
    df = pd.DataFrame(
        [
            {
                "event_id": "A:login:0",
                "customer_id": "A",
                "event_ts": T0,
                "event_type": "login",
                "value": 5.0,
            },
            {
                "event_id": "B:order:0",
                "customer_id": "B",
                "event_ts": T0 - pd.Timedelta(days=1),
                "event_type": "order",
                "value": math.nan,
            },
        ]
    )
    records = list(PROD.iter_records(df))
    assert len(records) == 2
    assert {r["customer_id"] for r in records} == {"A", "B"}
    assert all(set(PROD.EVENT_FIELDS) <= set(PROD.to_wire(r)) for r in records)
